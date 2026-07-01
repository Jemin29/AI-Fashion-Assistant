from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from week7.backend.configs.config import get_settings

logger = logging.getLogger("auth")
settings = get_settings()

# Define OAuth2 bearer extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a JWT access token containing standard subject claims and expiration."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.security.access_token_expire_minutes)

    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "type": "access"
    }
    encoded_jwt = jwt.encode(
        to_encode,
        settings.security.jwt_secret_key,
        algorithm=settings.security.jwt_algorithm
    )
    return encoded_jwt


def create_refresh_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a JWT refresh token with longer expiration lifetime."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Default to 7 days
        expire = datetime.now(timezone.utc) + timedelta(days=7)

    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "type": "refresh"
    }
    encoded_jwt = jwt.encode(
        to_encode,
        settings.security.jwt_secret_key,
        algorithm=settings.security.jwt_algorithm
    )
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """Decode a JWT token, verifying signatures and checking standard claims.

    Raises HTTPException on signature invalidation, mismatch, or expiration.
    """
    try:
        payload = jwt.decode(
            token,
            settings.security.jwt_secret_key,
            algorithms=[settings.security.jwt_algorithm]
        )
        return payload
    except JWTError as exc:
        logger.warning(f"Failed decoding JWT token: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> str:
    """FastAPI route dependency validating credentials. Returns the subject name."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)
    token_type = payload.get("type")
    
    # Refresh tokens are rejected on protected routes requiring access credentials
    if token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type for this endpoint",
            headers={"WWW-Authenticate": "Bearer"},
        )

    subject: Optional[str] = payload.get("sub")
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return subject
