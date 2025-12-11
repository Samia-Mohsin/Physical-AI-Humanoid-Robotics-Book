from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class User(Base):
    """
    User model for authentication and profile information
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)

    # Background questions from signup
    experience_level = Column(String)  # Beginner, Intermediate, Advanced
    has_rtx_gpu = Column(Boolean, default=False)
    has_jetson = Column(Boolean, default=False)
    preferred_language = Column(String, default="English")  # English, Urdu, Roman Urdu

    # Profile information
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True))


class Conversation(Base):
    """
    Conversation model for storing chat history
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # Foreign key to users table
    title = Column(String, nullable=False)  # Generated from first query
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Message(Base):
    """
    Message model for storing individual chat messages
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, index=True)  # Foreign key to conversations
    user_id = Column(Integer, index=True)  # Foreign key to users
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Additional metadata for RAG context
    selected_text = Column(Text)  # Text that was selected when message was sent
    sources = Column(JSON)  # List of source documents used in response


class Chapter(Base):
    """
    Chapter model for book content
    """
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(String, unique=True, index=True, nullable=False)  # e.g., "01-introduction"
    title = Column(String, nullable=False)
    content = Column(Text)  # Original English content
    word_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class TranslationCache(Base):
    """
    Translation cache model for storing translated content
    """
    __tablename__ = "translation_cache"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(String, index=True, nullable=False)  # Foreign key to chapters
    user_id = Column(Integer, index=True)  # Foreign key to users (for personalization)
    target_language = Column(String, nullable=False)  # "urdu", "roman_urdu"
    content_hash = Column(String, nullable=False)  # Hash of original content to detect changes
    translated_content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class PersonalizedContent(Base):
    """
    Personalized content model for storing personalized chapter versions
    """
    __tablename__ = "personalized_content"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(String, index=True, nullable=False)  # Foreign key to chapters
    user_id = Column(Integer, index=True, nullable=False)  # Foreign key to users
    user_experience_level = Column(String)  # To track which version was generated for which level
    has_rtx_gpu = Column(Boolean)
    has_jetson = Column(Boolean)
    personalized_content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Quiz(Base):
    """
    Quiz model for storing quiz questions per chapter
    """
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(String, index=True, nullable=False)  # Foreign key to chapters
    question = Column(Text, nullable=False)
    options = Column(JSON, nullable=False)  # List of options
    correct_answer = Column(String, nullable=False)
    explanation = Column(Text)  # Explanation of the correct answer
    difficulty = Column(String, default="medium")  # easy, medium, hard
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class QuizAttempt(Base):
    """
    Quiz attempt model for tracking user quiz performance
    """
    __tablename__ = "quiz_attempts"

    id = Column(Integer, primary_key=True, index=True)
    quiz_id = Column(Integer, index=True)  # Foreign key to quizzes
    user_id = Column(Integer, index=True)  # Foreign key to users
    chapter_id = Column(String, index=True)  # Foreign key to chapters
    selected_answer = Column(String)  # User's selected answer
    is_correct = Column(Boolean)  # Whether the answer was correct
    score = Column(Float)  # Score for the attempt (0.0 to 1.0)
    completed_at = Column(DateTime(timezone=True), server_default=func.now())


class DocumentChunk(Base):
    """
    Document chunk model for tracking which chunks were ingested into Qdrant
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(String, index=True, nullable=False)  # Foreign key to chapters
    chunk_id = Column(String, unique=True, nullable=False)  # ID used in Qdrant
    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False)  # To detect changes and re-ingest if needed
    chunk_metadata = Column(JSON)  # Additional metadata about the chunk (renamed from 'metadata' to avoid SQLAlchemy conflict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())