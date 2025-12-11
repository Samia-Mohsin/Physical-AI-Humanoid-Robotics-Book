from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta
import bcrypt
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from ..database import get_db
from ..models import User
import os
from dotenv import load_dotenv

load_dotenv()

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# No need for password context with direct bcrypt

# Security scheme
security = HTTPBearer()

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)

# Request models
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    # Background questions
    experience_level: Optional[str] = Field(None, description="Beginner, Intermediate, Advanced")
    has_rtx_gpu: Optional[bool] = Field(False, description="Has RTX GPU")
    has_jetson: Optional[bool] = Field(False, description="Has Jetson")
    preferred_language: Optional[str] = Field("English", description="Preferred language: English, Urdu, Roman Urdu")


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    experience_level: Optional[str]
    has_rtx_gpu: bool
    has_jetson: bool
    preferred_language: str
    created_at: datetime

    class Config:
        from_attributes = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """Hash a password"""
    # Truncate password to 72 bytes to comply with bcrypt limits
    if len(password) > 72:
        password = password[:72]
    # Convert password to bytes and hash it
    password_bytes = password.encode('utf-8')
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode('utf-8')


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email"""
    return db.query(User).filter(User.email == email).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password"""
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get the current user from the token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = get_user_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user


@router.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user with background questions"""
    # Check if user already exists
    existing_user = get_user_by_email(db, user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )

    # Hash the password
    hashed_password = get_password_hash(user.password)

    # Create new user
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        experience_level=user.experience_level,
        has_rtx_gpu=user.has_rtx_gpu,
        has_jetson=user.has_jetson,
        preferred_language=user.preferred_language
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@router.post("/login", response_model=Token)
async def login_user(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Login a user and return access token"""
    user = authenticate_user(db, user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login
    user.last_login_at = datetime.utcnow()
    db.commit()

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user's profile information"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        experience_level=current_user.experience_level,
        has_rtx_gpu=current_user.has_rtx_gpu,
        has_jetson=current_user.has_jetson,
        preferred_language=current_user.preferred_language,
        created_at=current_user.created_at
    )


@router.post("/logout")
async def logout_user():
    """Logout user (client-side token invalidation)"""
    # In a real implementation, you might add the token to a blacklist
    # For this implementation, we just tell the client to remove the token
    return {"message": "Successfully logged out"}


@router.put("/profile")
async def update_profile(
    user_update: UserCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile information"""
    # Update user fields
    current_user.full_name = user_update.full_name or current_user.full_name
    current_user.experience_level = user_update.experience_level or current_user.experience_level
    current_user.has_rtx_gpu = user_update.has_rtx_gpu or current_user.has_rtx_gpu
    current_user.has_jetson = user_update.has_jetson or current_user.has_jetson
    current_user.preferred_language = user_update.preferred_language or current_user.preferred_language

    db.commit()
    db.refresh(current_user)

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        experience_level=current_user.experience_level,
        has_rtx_gpu=current_user.has_rtx_gpu,
        has_jetson=current_user.has_jetson,
        preferred_language=current_user.preferred_language,
        created_at=current_user.created_at
    )