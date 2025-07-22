from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    Text,
    Integer,
    Float,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class AI_User(Base):
    __tablename__ = "ai_users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    chat_sessions = relationship(
        "AI_ChatSession", back_populates="user", cascade="all, delete-orphan"
    )
    documents = relationship("AI_Document", back_populates="admin", cascade="all, delete-orphan")

class AI_ChatSession(Base):
    __tablename__ = "ai_chat_sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, ForeignKey("ai_users.email"), nullable=False, index=True)
    title = Column(String, default="New Consultation")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    user = relationship("AI_User", back_populates="chat_sessions")
    messages = relationship(
        "AI_ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )

class AI_ChatMessage(Base):
    __tablename__ = "ai_chat_messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(
        String, ForeignKey("ai_chat_sessions.id"), nullable=False, index=True
    )
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    references = Column(Text)
    session = relationship("AI_ChatSession", back_populates="messages")

class AI_Document(Base):
    __tablename__ = "ai_documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    admin_email = Column(
        String,
        ForeignKey("ai_users.email", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_on = Column(DateTime, default=datetime.utcnow, index=True)
    processed = Column(Boolean, default=False)
    page_count = Column(Integer, default=0)
    is_public = Column(Boolean, default=True)
    author = Column(String)
    description = Column(Text)
    category = Column(String)
    admin = relationship("AI_User", back_populates="documents")
    chunks = relationship(
        "AI_DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

class AI_DocumentChunk(Base):
    __tablename__ = "ai_document_chunks"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("ai_documents.id", ondelete="CASCADE"))
    chunk_text = Column(Text, nullable=False)
    page_number = Column(Integer)
    chunk_index = Column(Integer)
    embedding_id = Column(String, unique=True)
    chapter_name = Column(String)
    section_name = Column(String)
    category = Column(String)
    relevance_score = Column(Float)
    last_accessed = Column(DateTime)
    document = relationship("AI_Document", back_populates="chunks") 