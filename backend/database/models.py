from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

# 1. Clients Table (Lead + Client tracker)
class Client(Base):
    __tablename__ = "clients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    investment_profile = Column(String)
    status = Column(String, default="Lead") # Track if they are a Lead, Active, or At-Risk
    
    portfolios = relationship("Portfolio", back_populates="owner")
    meetings = relationship("Meeting", back_populates="client")
    invoices = relationship("Invoice", back_populates="client")

# 2. Portfolios Table
class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    assets = Column(String)
    value = Column(Float)
    risk_score = Column(Float)
    
    owner = relationship("Client", back_populates="portfolios")

# 3. Services Table (Your Product Catalog)
class Service(Base):
    __tablename__ = "services"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    pricing = Column(String)

# 4. Meetings Table
class Meeting(Base):
    __tablename__ = "meetings"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    datetime = Column(DateTime, default=datetime.utcnow)
    advisor = Column(String)
    
    client = relationship("Client", back_populates="meetings")

# 5. Invoices Table (NEW: For Cash Flow tracking)
class Invoice(Base):
    __tablename__ = "invoices"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    amount = Column(Float)
    is_paid = Column(Boolean, default=False)
    due_date = Column(DateTime)

    client = relationship("Client", back_populates="invoices")

# 6. Users Table
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="admin")