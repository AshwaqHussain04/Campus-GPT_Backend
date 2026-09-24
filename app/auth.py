from datetime import datetime, timedelta, timezone
from typing import Literal

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from app.db import get_admin_for_login, get_student_for_login
from app.config import settings

bearer_scheme = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def authenticate_admin(username: str, password: str) -> dict | bool:
    user = get_admin_for_login(username)
    if user:
        return user if verify_password(password, user["hashed_password"]) else False

    # Allows the configured bootstrap admin to create the first database admin.
    if username == settings.admin_username:
        return verify_password(password, settings.admin_password_hash)
    return False


def authenticate_student(email: str, password: str) -> dict | bool:
    user = get_student_for_login(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(subject: str, role: Literal["student", "admin"]) -> str:
    if role not in {"student", "admin"}:
        raise ValueError("Unsupported token role")
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        role = payload.get("role")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
        if role != "admin":
            raise HTTPException(status_code=403, detail="Admin authentication required")

        return username
    
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def get_current_student(
        credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        email: str = payload.get("sub")
        role = payload.get("role")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
       
        if role != "student":
            raise HTTPException(status_code=403, detail="Student authentication required")

        return email
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")