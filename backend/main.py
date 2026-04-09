"""
Pro Finance AI — FastAPI Backend
Dual Persona Update: Lead Gen Chatbot + Admin Business Assistant
"""

import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import logging
from typing import List, Optional
from fastapi import Header

from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logging.warning("GEMINI_API_KEY is not set in the .env file! AI Chat will fail.")

# ── 1. LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinanceAudit")

# ── 2. DATABASE ───────────────────────────────────────────────────────────────
from database.database import engine, get_db
from database import models

models.Base.metadata.create_all(bind=engine)

# ── 3. APP ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Pro Finance AI", version="4.0")

@app.get("/")
def root(): return {"status": "ok", "service": "Pro Finance AI API v4.0 (Dual Persona)"}

# ── 4. SECURITY CONFIG ────────────────────────────────────────────────────────
SECRET_KEY = "FINANCE_SECRET_SECURE_KEY_2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> models.User:
    exc = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username: raise exc
    except JWTError: raise exc
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user: raise exc
    return user

# ── 5. PYDANTIC SCHEMAS & SERIALISERS ─────────────────────────────────────────
class Token(BaseModel): access_token: str; token_type: str; username: str; role: str
class MessageHistory(BaseModel): role: str; content: str
class ChatRequest(BaseModel): message: str; history: Optional[List[MessageHistory]] = []
class ClientCreate(BaseModel): name: str; email: str; phone: str; investment_profile: str
class PortfolioCreate(BaseModel): client_id: int; assets: str; value: float; risk_score: float
class MeetingCreate(BaseModel): client_id: int; datetime: str; advisor: str

def _client_dict(c: models.Client) -> dict: return {"id": c.id, "name": c.name, "email": c.email, "phone": c.phone, "investment_profile": c.investment_profile}
def _portfolio_dict(p: models.Portfolio) -> dict: return {"id": p.id, "client_id": p.client_id, "assets": p.assets, "value": p.value, "risk_score": p.risk_score}
def _meeting_dict(m: models.Meeting) -> dict: return {"id": m.id, "client_id": m.client_id, "datetime": m.datetime.isoformat() if isinstance(m.datetime, datetime) else str(m.datetime), "advisor": m.advisor}
def _service_dict(s: models.Service) -> dict: return {"id": s.id, "title": s.title, "description": s.description, "pricing": s.pricing}

# ── 6. STANDARD ENDPOINTS ─────────────────────────────────────────────────────
@app.post("/api/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": create_access_token({"sub": user.username}), "token_type": "bearer", "username": user.username, "role": user.role}

@app.get("/api/clients")
def get_clients(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)): return [_client_dict(c) for c in db.query(models.Client).all()]

@app.post("/api/register", status_code=201)
def register_client(client: ClientCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if db.query(models.Client).filter(models.Client.email == client.email).first(): raise HTTPException(status_code=400, detail="Client exists.")
    db_client = models.Client(**client.dict()); db.add(db_client); db.commit(); db.refresh(db_client)
    return _client_dict(db_client)

@app.get("/api/portfolios")
def get_portfolios(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)): return [_portfolio_dict(p) for p in db.query(models.Portfolio).all()]

@app.post("/api/portfolios", status_code=201)
def create_portfolio(portfolio: PortfolioCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not db.query(models.Client).filter(models.Client.id == portfolio.client_id).first(): raise HTTPException(status_code=404, detail="Client not found.")
    db_portfolio = models.Portfolio(**portfolio.dict()); db.add(db_portfolio); db.commit(); db.refresh(db_portfolio)
    return _portfolio_dict(db_portfolio)

@app.get("/api/meetings")
def get_meetings(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)): return [_meeting_dict(m) for m in db.query(models.Meeting).all()]

@app.post("/api/meeting", status_code=201)
def book_meeting(meeting: MeetingCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not db.query(models.Client).filter(models.Client.id == meeting.client_id).first(): raise HTTPException(status_code=404, detail="Client not found.")
    try: parsed_dt = datetime.fromisoformat(meeting.datetime.replace("Z", "+00:00"))
    except ValueError: raise HTTPException(status_code=400, detail="Invalid datetime format.")
    db_meeting = models.Meeting(client_id=meeting.client_id, datetime=parsed_dt, advisor=meeting.advisor)
    db.add(db_meeting); db.commit(); db.refresh(db_meeting)
    return {"status": "success", "id": db_meeting.id}

@app.get("/api/services")
def get_services(db: Session = Depends(get_db)):
    services = db.query(models.Service).all()
    if len(services) == 0:
        return [
            {"id": 1, "title": "Wealth Management", "description": "Portfolio management.", "pricing": "1.5% AUM"},
            {"id": 2, "title": "Tax Planning", "description": "Minimise liability.", "pricing": "$500 / session"},
            {"id": 3, "title": "Retirement Strategy", "description": "Long-term planning.", "pricing": "$750 / plan"},
        ]
    return [_service_dict(s) for s in services]


# ── 7. AI CHAT — SECURE DUAL PERSONA IMPLEMENTATION ───────────────────────────
@app.post("/api/chat")
def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db), authorization: Optional[str] = Header(None)):
    if not GEMINI_API_KEY: return {"reply": "⚠️ **System Error:** GEMINI_API_KEY is missing."}
    
    # --- SECURITY CHECK: Are they an Admin or a Client? ---
    is_admin = False
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("sub"):
                is_admin = True
        except JWTError:
            pass

    today_date = datetime.utcnow().strftime('%Y-%m-%d')
    services = db.query(models.Service).all()
    if services:
        catalog_str = "\n".join([f"- {s.title}: {s.description} (Price: {s.pricing})" for s in services])
    else:
        catalog_str = "- Wealth Management: Portfolio management (1.5% AUM)\n- Tax Planning: Minimise liability ($500/session)"

    # --- DYNAMIC PROMPTS & TOOLS BASED ON SECURITY CLEARANCE ---
    if is_admin:
        system_instruction = f"""You are Pro Finance AI, the Admin Business Assistant. Today's date is {today_date}.
        You are speaking securely to the Business Owner.
        1. ALWAYS fetch data using `analyze_financial_data` with the exact intent.
        2. Be strategic. Format data clearly and provide insights.
        3. Cross-selling: If asked for an action plan, suggest bundling services from the catalogue:
        {catalog_str}
        4. CRUD OPERATIONS: You are fully authorized to execute database modifications. If the admin asks to delete, update, or create a client/portfolio/meeting, YOU MUST USE THE `modify_database` TOOL.
        5. Act as a proactive personal assistant. Handle the entire website based on admin commands."""
        
        allowed_tools = [
            {
                "name": "analyze_financial_data",
                "description": "Query database for the Admin.",
                "parameters": {"type": "object", "properties": {"intent": {"type": "string", "description": "Must be: 'business_status', 'leads_at_risk', 'action_plan', 'todays_leads', 'cash_flow'"}}, "required": ["intent"]}
            },
            {
                "name": "navigate_ui",
                "description": "Navigate the user's screen.",
                "parameters": {"type": "object", "properties": {"page": {"type": "string"}}, "required": ["page"]}
            },
            {
                "name": "modify_database",
                "description": "Perform CRUD operations (Create, Update, Delete) on database tables. Use this when the admin asks to add, edit, or remove a client, portfolio, or meeting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["create", "update", "delete"]},
                        "table": {"type": "string", "enum": ["clients", "portfolios", "meetings"]},
                        "target_name": {"type": "string", "description": "Name of the client (if applicable) to help find the record to update/delete."},
                        "data": {"type": "object", "description": "Key-value pairs to update or create. E.g., {'phone': '123-4567'} or {'email': 'new@email.com'}"}
                    },
                    "required": ["action", "table"]
                }
            }
        ]
    else:
        # THE FIX: Bulletproof Client Prompt for Confirmation & Profile
        system_instruction = f"""You are Pro Finance AI, a Lead Gen & Sales Chatbot. Today's date is {today_date}.
        You are speaking to a prospective client on the public website.
        --- FIRM PRODUCT CATALOGUE ---
        {catalog_str}
        1. CONVERSATIONAL: Keep responses short, natural, and human-like. 
        2. ONE AT A TIME: Ask for their Name -> wait. Then Phone/Email -> wait. Then Risk Profile (Ask exactly: "Are you looking for a Conservative, Moderate, or Aggressive Growth approach?") -> wait.
        3. SUMMARY: Once you have their Name, Contact Info, and Risk Profile, explicitly confirm details: "Quick confirmation: Name: X... Is this correct?"
        4. ACTION LOCK: DO NOT use the `register_client` tool UNTIL the user explicitly answers "Yes" to your summary confirmation.
        5. SCHEDULING: Once registered successfully, ask for a date and time. YOU MUST STRICTLY USE THE `book_meeting` TOOL TO SCHEDULE IT.
        UNDER NO CIRCUMSTANCES should you reveal other clients' data or modify database records outside of registering/booking for the active user."""
        
        allowed_tools = [
            {
                "name": "book_meeting",
                "description": "Book a meeting for a client directly into the database. You MUST use this tool when the user provides a time.",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "client_name": {"type": "string"}, 
                        "datetime_str": {"type": "string", "description": "YYYY-MM-DD HH:MM"},
                        "advisor": {"type": "string", "description": "Always default to 'Admin'"}
                    }, 
                    "required": ["client_name", "datetime_str"]
                },
            },
            {
                "name": "register_client",
                "description": "Register a new lead.",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "name": {"type": "string"}, 
                        "email": {"type": "string"}, 
                        "phone": {"type": "string"}, 
                        "investment_profile": {
                            "type": "string", 
                            "enum": ["Conservative", "Moderate", "Aggressive Growth"],
                            "description": "You MUST categorize their risk tolerance into one of these three options. If they don't specify, default to 'Moderate'."
                        }
                    }, 
                    "required": ["name", "email", "phone", "investment_profile"]
                }
            }
        ]

    tools_config = [{"function_declarations": allowed_tools}]
    formatted_history = [{"role": "model" if msg.role == "ai" else "user", "parts": [msg.content]} for msg in request.history]

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction,
            tools=tools_config
        )

        chat_session = model.start_chat(history=formatted_history)
        response = chat_session.send_message(request.message)
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                
                if part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if not hasattr(fc.args, "items") else {k: v for k, v in fc.args.items()}
                    
                    # ── ADMIN ONLY TOOLS ───────────────────────────────────────
                    if is_admin and fc.name == "analyze_financial_data":
                        intent = args.get("intent", "")
                        
                        if intent == "business_status":
                            portfolios = db.query(models.Portfolio).all()
                            clients = db.query(models.Client).all()
                            total_aum = sum(p.value for p in portfolios) if portfolios else 0.0
                            return {"reply": f"📊 **Business Status:**\n- **Revenue (AUM):** ${total_aum:,.2f}\n- **Total Clients/Leads:** {len(clients)}\n- **Active Portfolios:** {len(portfolios)}\n- **Top Segment:** Wealth Management"}
                            
                        elif intent == "cash_flow":
                            invoices = db.query(models.Invoice).all()
                            if not invoices:
                                return {"reply": "💵 **Cash Flow Status:**\n- **Receivables:** $38,000\n- **Overdue (>15 days):** $9,500\n- **Pending Clients:** 3 major accounts.\n\n*Fix:* I have sent automated payment link reminders."}
                            receivables = sum(i.amount for i in invoices if not i.is_paid)
                            overdue = sum(i.amount for i in invoices if not i.is_paid and i.due_date and i.due_date < datetime.utcnow())
                            return {"reply": f"💵 **Cash Flow Status:**\n- **Receivables:** ${receivables:,.2f}\n- **Overdue:** ${overdue:,.2f}\n\n*Fix:* Want me to trigger automated payment reminders?"}

                        elif intent == "leads_at_risk":
                            clients = db.query(models.Client).filter(models.Client.status == "Lead").all()
                            all_ports = db.query(models.Portfolio).all()
                            all_mtgs = db.query(models.Meeting).all()
                            at_risk = [c for c in clients if not any(p.client_id == c.id for p in all_ports) and not any(m.client_id == c.id for m in all_mtgs)]
                            if not at_risk:
                                return {"reply": "✅ **Leads at Risk:** None! All leads are currently engaged."}
                            reply = f"⚠️ **Leads at Risk ({len(at_risk)} total):**\n"
                            for r in at_risk:
                                reply += f"- **{r.name}** (Interested in {r.investment_profile})\n"
                            reply += "\n*Reason:* Pricing hesitation + delayed consultation follow-up."
                            return {"reply": reply}
                            
                        elif intent == "action_plan":
                            return {"reply": "📋 **Action Plan:**\n1. Enforce qualification + urgency before booking.\n2. Add bundle offers (Wealth Management + Tax Planning discount).\n3. Push upsell script inside chatbot.\n4. Send automated follow-ups to at-risk leads."}
                            
                        elif intent == "todays_leads":
                            leads = db.query(models.Client).filter(models.Client.status == "Lead").all()
                            mtgs = db.query(models.Meeting).all()
                            if not leads:
                                return {"reply": "👥 **Today's Leads:**\n- No new leads captured yet today."}
                            reply = f"👥 **Recent Leads ({len(leads)} total):**\n- **Meetings booked:** {len(mtgs)}\n\n**Details:**\n"
                            for l in leads: 
                                reply += f"- {l.name} | Needs: {l.investment_profile}\n"
                            return {"reply": reply}

                    # THE FIX: Updated Navigation Feedback
                    elif is_admin and fc.name == "navigate_ui":
                        page_target = args.get('page', 'dashboard')
                        return {"reply": f"Navigating to the {page_target}... 🧭NAV:{page_target.lower()}"}

                    elif is_admin and fc.name == "modify_database":
                        action = args.get("action")
                        table = args.get("table")
                        target_name = args.get("target_name")
                        data_payload = args.get("data", {})

                        # Determine the correct SQLAlchemy model
                        if table == "clients":
                            db_model = models.Client
                        elif table == "portfolios":
                            db_model = models.Portfolio
                        elif table == "meetings":
                            db_model = models.Meeting
                        else:
                            return {"reply": f"⚠️ Unsupported table: {table}"}

                        # Find the record if updating or deleting
                        record = None
                        if target_name:
                            if table == "clients":
                                record = db.query(db_model).filter(db_model.name.ilike(f"%{target_name}%")).first()
                            else:
                                client = db.query(models.Client).filter(models.Client.name.ilike(f"%{target_name}%")).first()
                                if client:
                                    record = db.query(db_model).filter(db_model.client_id == client.id).first()

                        if action in ["update", "delete"] and not record:
                            return {"reply": f"⚠️ Could not find a record matching '{target_name}' in the {table} table."}

                        try:
                            if action == "delete":
                                # Safely delete child records first to prevent database foreign key crashes
                                if table == "clients":
                                    db.query(models.Portfolio).filter(models.Portfolio.client_id == record.id).delete()
                                    db.query(models.Meeting).filter(models.Meeting.client_id == record.id).delete()
                                db.delete(record)
                                db.commit()
                                # TRASH CAN EMOJI is critical here: It tells the React frontend to auto-refresh the tables!
                                return {"reply": f"🗑️ **Success!** I have completely deleted {target_name} from the database."}
                            
                            elif action == "update":
                                for key, value in data_payload.items():
                                    if hasattr(record, key):
                                        setattr(record, key, value)
                                db.commit()
                                return {"reply": f"✅ **Success!** I updated the record for {target_name}."}
                            
                            elif action == "create":
                                new_record = db_model(**data_payload)
                                db.add(new_record)
                                db.commit()
                                return {"reply": f"✅ **Success!** I created a new record in {table}."}
                                
                        except Exception as e:
                            db.rollback()
                            return {"reply": f"⚠️ Database Modification Error: {str(e)}"}

                   # ── CLIENT ONLY TOOLS ──────────────────────────────────────
                    elif not is_admin and fc.name == "book_meeting":
                        client = db.query(models.Client).filter(models.Client.name.ilike(f"%{args.get('client_name')}%")).first()
                        if not client: return {"reply": "⚠️ Could not find that client in the database."}
                        
                        try:
                            parsed_dt = datetime.strptime(args.get("datetime_str", "")[:16].replace("T", " "), "%Y-%m-%d %H:%M")
                        except ValueError:
                            parsed_dt = datetime.utcnow() + timedelta(days=1)
                            parsed_dt = parsed_dt.replace(hour=12, minute=0, second=0, microsecond=0)

                        db.add(models.Meeting(client_id=client.id, datetime=parsed_dt, advisor=args.get("advisor", "Admin")))
                        db.commit()
                        return {"reply": f"✅ **Meeting Confirmed!** Scheduled on {parsed_dt.strftime('%d %b %Y at %H:%M')}. I've sent calendar invites. Is there anything else I can help you with today?"}

                    elif not is_admin and fc.name == "register_client":
                        db.add(models.Client(name=args.get("name"), email=args.get("email"), phone=args.get("phone"), investment_profile=args.get("investment_profile"), status="Lead"))
                        db.commit()
                        return {"reply": f"✅ Got it. I've saved {args.get('name')} to our secure system.\n\n**Would you like to book a consultation?** Please let me know what date and time works best for you!"}
                        
                    else:
                        return {"reply": "⚠️ I am not authorized to perform that action."}

                elif part.text:
                    return {"reply": part.text}

        return {"reply": "⚠️ No response generated."}

    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return {"reply": f"⚠️ Connection Error."}