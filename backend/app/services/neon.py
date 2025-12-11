from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from ..models import User, Conversation, Message
from ..database import get_db
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeonService:
    """Service class for Neon Postgres database operations using SQLAlchemy"""

    def __init__(self):
        logger.info("NeonService initialized")

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile information from Neon Postgres
        """
        try:
            # Since user_id is stored as integer in the database but passed as string in API,
            # we need to convert it
            user_id_int = int(user_id) if user_id.isdigit() else None

            if not user_id_int:
                logger.error(f"Invalid user_id format: {user_id}")
                return None

            from ..database import SessionLocal
            db = SessionLocal()

            try:
                user = db.query(User).filter(User.id == user_id_int).first()
                if user:
                    return {
                        "id": user.id,
                        "email": user.email,
                        "full_name": user.full_name,
                        "experience_level": user.experience_level,
                        "has_rtx_gpu": user.has_rtx_gpu,
                        "has_jetson": user.has_jetson,
                        "preferred_language": user.preferred_language
                    }
                return None
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error getting user profile: {str(e)}")
            return None

    def save_conversation(self, user_id: str, title: str) -> Optional[int]:
        """
        Save a new conversation to Neon Postgres
        """
        try:
            user_id_int = int(user_id) if user_id and user_id.isdigit() else None

            if not user_id_int:
                logger.error(f"Invalid user_id format: {user_id}")
                return None

            from ..database import SessionLocal
            db = SessionLocal()

            try:
                conversation = Conversation(
                    user_id=user_id_int,
                    title=title
                )
                db.add(conversation)
                db.commit()
                db.refresh(conversation)

                return conversation.id
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            return None

    def save_message(self, conversation_id: int, user_id: str, role: str, content: str, selected_text: Optional[str] = None) -> bool:
        """
        Save a message to Neon Postgres
        """
        try:
            user_id_int = int(user_id) if user_id and user_id.isdigit() else None

            if not user_id_int:
                logger.error(f"Invalid user_id format: {user_id}")
                return False

            from ..database import SessionLocal
            db = SessionLocal()

            try:
                message = Message(
                    conversation_id=conversation_id,
                    user_id=user_id_int,
                    role=role,
                    content=content,
                    selected_text=selected_text
                )
                db.add(message)
                db.commit()

                return True
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            return False

    def get_conversation_history(self, conversation_id: int) -> list:
        """
        Retrieve conversation history from Neon Postgres
        """
        try:
            from ..database import SessionLocal
            db = SessionLocal()

            try:
                messages = db.query(Message).filter(
                    Message.conversation_id == conversation_id
                ).order_by(Message.created_at).all()

                return [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                        "selected_text": msg.selected_text
                    }
                    for msg in messages
                ]
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []

    def get_user_conversations(self, user_id: str) -> list:
        """
        Retrieve all conversations for a specific user
        """
        try:
            user_id_int = int(user_id) if user_id and user_id.isdigit() else None

            if not user_id_int:
                logger.error(f"Invalid user_id format: {user_id}")
                return []

            from ..database import SessionLocal
            db = SessionLocal()

            try:
                conversations = db.query(Conversation).filter(
                    Conversation.user_id == user_id_int
                ).order_by(Conversation.created_at.desc()).all()

                return [
                    {
                        "id": conv.id,
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat() if conv.created_at else None,
                        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None
                    }
                    for conv in conversations
                ]
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error getting user conversations: {str(e)}")
            return []

# Global instance of NeonService
neon_service = NeonService()

def get_neon_service() -> NeonService:
    """Get the global instance of NeonService"""
    return neon_service