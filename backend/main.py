"""
Pro Finance AI — FastAPI Backend
All endpoints tested against models.py schema.
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
app = FastAPI(title="Pro Finance AI", version="3.0")

@app.get("/")
def root(): return {"status": "ok", "service": "Pro Finance AI API v3.0"}

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
class ChatRequest(BaseModel): message: str
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

@app.get("/api/analytics")
def get_analytics(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    clients = db.query(models.Client).all(); portfolios = db.query(models.Portfolio).all(); meetings = db.query(models.Meeting).all()
    total_aum = sum(p.value for p in portfolios); avg_risk = (sum(p.risk_score for p in portfolios) / len(portfolios)) if portfolios else 0
    return {"total_aum": f"${total_aum:,.2f}", "total_clients": len(clients), "total_meetings": len(meetings), "average_risk_score": round(avg_risk, 2), "portfolio_count": len(portfolios)}


# ── 7. AI CHAT — GEMINI WITH FULL CRUD AND ROUTING ────────────────────────────
@app.post("/api/chat")
def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    if not GEMINI_API_KEY: return {"reply": "⚠️ **System Error:** GEMINI_API_KEY is missing."}

    clients = db.query(models.Client).all()
    portfolios = db.query(models.Portfolio).all()
    meetings = db.query(models.Meeting).all()
    today_date = datetime.utcnow().strftime('%Y-%m-%d')

    client_lines = []
    for c in clients:
        c_portfolios = [p for p in portfolios if p.client_id == c.id]
        c_meetings = [m for m in meetings if m.client_id == c.id]
        c_aum = sum(p.value for p in c_portfolios)
        c_mtgs = "; ".join([m.datetime.strftime("%d %b %Y %H:%M") for m in c_meetings]) or "None"
        client_lines.append(f"• {c.name} | Email: {c.email} | Phone: {c.phone} | Profile: {c.investment_profile} | AUM: ${c_aum:,.2f} | Meetings: {c_mtgs}")

    client_block = "\n".join(client_lines) if client_lines else "(No clients yet)"

    system_instruction = f"""You are Pro Finance AI, an expert wealth management assistant.
    Today's date is {today_date}.
    
    FIRM DATABASE:
    {client_block}
    
    RULES:
    1. Answer finance questions expertly.
    2. Read data directly from the FIRM DATABASE.
    3. Use tools to Create, Update, or Delete records when requested.
    4. STRICT BOOKING RULE: If the user asks to book a meeting but does NOT specify exact date, time, and advisor, DO NOT use the tool yet. Ask them to clarify the missing details.
    5. UI NAVIGATION: If the user asks to go to, navigate to, or asks 'how to go to' a page (like clients, portfolios, dashboard), you MUST use the `Maps_ui` tool to take them there directly!
    """

    book_meeting_tool = {"function_declarations": [{"name": "book_meeting", "description": "Book a meeting.", "parameters": {"type": "object", "properties": {"client_name": {"type": "string"}, "datetime_str": {"type": "string", "description": "YYYY-MM-DD HH:MM"}, "advisor": {"type": "string"}}, "required": ["client_name", "datetime_str", "advisor"]}}]}
    register_client_tool = {"function_declarations": [{"name": "register_client", "description": "Register a new client.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"}, "investment_profile": {"type": "string"}}, "required": ["name", "email", "phone", "investment_profile"]}}]}
    create_portfolio_tool = {"function_declarations": [{"name": "create_portfolio", "description": "Create portfolio.", "parameters": {"type": "object", "properties": {"client_name": {"type": "string"}, "assets": {"type": "string", "description": "Calculate allocation based on risk_score (1-10). e.g., '90% Stocks, 10% Bonds'"}, "value": {"type": "number"}, "risk_score": {"type": "number"}}, "required": ["client_name", "assets", "value", "risk_score"]}}]}
    update_client_tool = {"function_declarations": [{"name": "update_client", "description": "Update an existing client's details.", "parameters": {"type": "object", "properties": {"client_name": {"type": "string"}, "new_email": {"type": "string"}, "new_phone": {"type": "string"}, "new_profile": {"type": "string"}}, "required": ["client_name"]}}]}
    delete_record_tool = {"function_declarations": [{"name": "delete_record", "description": "Delete a client, meeting, or portfolio.", "parameters": {"type": "object", "properties": {"client_name": {"type": "string"}, "record_type": {"type": "string", "description": "Must be 'Client', 'Portfolio', or 'Meeting'"}}, "required": ["client_name", "record_type"]}}]}
    
    # NEW TOOL: UI Navigation
    navigate_ui_tool = {"function_declarations": [{"name": "navigate_ui", "description": "Navigate the user's screen to a specific page.", "parameters": {"type": "object", "properties": {"page": {"type": "string", "description": "Allowed values: dashboard, clients, portfolios, services, meetings"}}, "required": ["page"]}}]}

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction,
            tools=[book_meeting_tool, register_client_tool, create_portfolio_tool, update_client_tool, delete_record_tool, navigate_ui_tool]
        )

        response = model.generate_content(request.message)
        
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            
            if part.function_call:
                fc = part.function_call
                args = fc.args
                
                # --- NEW: Routing Action ---
                if fc.name == "navigate_ui":
                    page = args.get("page", "dashboard").lower()
                    return {"reply": f"Taking you to the {page.capitalize()} page! 🧭NAV:{page}"}

                # --- Database Actions ---
                client_name = args.get("client_name", "")
                client = db.query(models.Client).filter(models.Client.name.ilike(f"%{client_name}%")).first()
                if not client and " " in client_name:
                    client = db.query(models.Client).filter(models.Client.name.ilike(f"%{client_name.split()[0]}%")).first()

                if fc.name == "book_meeting":
                    if not client: return {"reply": f"⚠️ Could not find client '{client_name}'."}
                    try: parsed_dt = datetime.strptime(args.get("datetime_str")[:16].replace("T", " "), "%Y-%m-%d %H:%M")
                    except ValueError: return {"reply": "⚠️ I couldn't understand the time format."}
                    db.add(models.Meeting(client_id=client.id, datetime=parsed_dt, advisor=args.get("advisor", "Admin")))
                    db.commit()
                    return {"reply": f"✅ **Meeting Booked!** Scheduled for {client.name} on {parsed_dt.strftime('%d %b %Y at %H:%M')}."}

                elif fc.name == "register_client":
                    if db.query(models.Client).filter(models.Client.email == args.get("email")).first():
                        return {"reply": f"⚠️ A client with email {args.get('email')} already exists."}
                    db.add(models.Client(name=args.get("name"), email=args.get("email"), phone=args.get("phone"), investment_profile=args.get("investment_profile")))
                    db.commit()
                    return {"reply": f"✅ **Client Registered!** Added {args.get('name')} to the database."}

                elif fc.name == "create_portfolio":
                    if not client: return {"reply": f"⚠️ Could not find client '{client_name}'."}
                    db.add(models.Portfolio(client_id=client.id, assets=args.get("assets"), value=float(args.get("value")), risk_score=float(args.get("risk_score"))))
                    db.commit()
                    return {"reply": f"✅ **Portfolio Created!** Added to {client.name}'s profile."}

                elif fc.name == "update_client":
                    if not client: return {"reply": f"⚠️ Could not find client '{client_name}'."}
                    if args.get("new_email"): client.email = args.get("new_email")
                    if args.get("new_phone"): client.phone = args.get("new_phone")
                    if args.get("new_profile"): client.investment_profile = args.get("new_profile")
                    db.commit()
                    return {"reply": f"✅ **Client Updated!** Modifications saved for {client.name}."}

                elif fc.name == "delete_record":
                    if not client: return {"reply": f"⚠️ Could not find client '{client_name}'."}
                    r_type = args.get("record_type")
                    
                    if r_type == "Client":
                        # FIX: Cascading Deletes - Delete orphaned records first!
                        db.query(models.Portfolio).filter(models.Portfolio.client_id == client.id).delete()
                        db.query(models.Meeting).filter(models.Meeting.client_id == client.id).delete()
                        db.delete(client)
                        db.commit()
                        return {"reply": f"🗑️ **Client Deleted!** Removed {client.name} and all related portfolios/meetings."}
                    
                    elif r_type == "Portfolio":
                        port = db.query(models.Portfolio).filter(models.Portfolio.client_id == client.id).first()
                        if port: db.delete(port); db.commit(); return {"reply": f"🗑️ **Portfolio Deleted!** Removed portfolio for {client.name}."}
                        return {"reply": f"⚠️ {client.name} does not have a portfolio to delete."}
                    
                    elif r_type == "Meeting":
                        mtg = db.query(models.Meeting).filter(models.Meeting.client_id == client.id).first()
                        if mtg: db.delete(mtg); db.commit(); return {"reply": f"🗑️ **Meeting Deleted!** Canceled the meeting for {client.name}."}
                        return {"reply": f"⚠️ {client.name} does not have any upcoming meetings."}

            if part.text:
                return {"reply": part.text}

        return {"reply": "⚠️ I processed your request, but generated no text response."}

    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return {"reply": f"⚠️ **Connection Error:** Failed to connect to Google Gemini API."}