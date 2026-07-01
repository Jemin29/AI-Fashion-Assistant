from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from week7.backend.configs.rate_limit import limiter

from week7.backend.auth.security import get_password_hash, verify_password
from week7.backend.auth.jwt_auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Inline thread-safe in-memory database to store registered users for validation
_USERS_DB: Dict[str, str] = {}


class RegisterSchema(BaseModel):
    username: str = Field(..., description="Unique alphanumeric username")
    password: str = Field(..., description="Secure plaintext password")


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="Valid JWT refresh token")


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


@router.post("/register", status_code=status.HTTP_201_CREATED)
@limiter.limit("15/minute")
def register_user(request: Request, payload: RegisterSchema) -> Dict[str, str]:
    """Register a new username and store its secure hashed password."""
    username = payload.username.strip()
    if not username or not payload.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password cannot be empty."
        )

    if username in _USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already registered."
        )

    # Hash the password and save
    hashed = get_password_hash(payload.password)
    _USERS_DB[username] = hashed
    return {"message": "User registered successfully", "username": username}


@router.post("/token", response_model=TokenResponse)
@limiter.limit("15/minute")
def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
) -> TokenResponse:
    """Exchange OAuth2 form username/password credentials for access and refresh tokens."""
    username = form_data.username.strip()
    hashed = _USERS_DB.get(username)

    if not hashed or not verify_password(form_data.password, hashed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Issue tokens
    access = create_access_token(username)
    refresh = create_refresh_token(username)

    return TokenResponse(
        access_token=access,
        refresh_token=refresh
    )


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("15/minute")
def refresh_access_token(request: Request, payload: RefreshRequest) -> TokenResponse:
    """Validate a refresh token and issue a brand-new access token pair."""
    token_data = decode_token(payload.refresh_token)
    
    # Confirm token is indeed a refresh token
    if token_data.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type for refresh endpoint",
        )

    username = token_data.get("sub")
    if not username or username not in _USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or credentials invalid",
        )

    # Re-issue new token pair
    access = create_access_token(username)
    refresh = create_refresh_token(username)

    return TokenResponse(
        access_token=access,
        refresh_token=refresh
    )


@router.get("/me")
def get_current_user_profile(username: str = Depends(get_current_user)) -> Dict[str, str]:
    """Protected route returning the profile details of the currently authenticated user."""
    return {
        "username": username,
        "role": "designer",
        "status": "active"
    }
