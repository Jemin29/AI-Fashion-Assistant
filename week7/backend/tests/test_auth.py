from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from week7.backend.main import app
from week7.backend.configs.config import get_settings

client = TestClient(app)
settings = get_settings()


def test_auth_full_lifecycle():
    """Verify register, login, refresh, and profile fetching protected route access."""
    username = "fashion_designer_test"
    password = "SuperSecurePassword123"

    # 1. Access protected route without credentials -> 401
    me_resp_anonymous = client.get("/auth/me")
    assert me_resp_anonymous.status_code == 401

    # 2. Register user -> 201
    reg_resp = client.post("/auth/register", json={
        "username": username,
        "password": password
    })
    assert reg_resp.status_code == 201
    assert reg_resp.json()["username"] == username

    # 3. Duplicate registration -> 400
    dup_resp = client.post("/auth/register", json={
        "username": username,
        "password": password
    })
    assert dup_resp.status_code == 400

    # 4. Login with invalid password -> 401
    bad_login = client.post("/auth/token", data={
        "username": username,
        "password": "WrongPassword"
    })
    assert bad_login.status_code == 401

    # 5. Login with correct credentials -> 200 and returns tokens
    login_resp = client.post("/auth/token", data={
        "username": username,
        "password": password
    })
    assert login_resp.status_code == 200
    tokens = login_resp.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["token_type"] == "bearer"

    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    # 6. Access protected route with valid token -> 200
    headers = {"Authorization": f"Bearer {access_token}"}
    me_resp = client.get("/auth/me", headers=headers)
    assert me_resp.status_code == 200
    profile = me_resp.json()
    assert profile["username"] == username
    assert profile["role"] == "designer"

    # 7. Refresh token flow -> 200
    refresh_resp = client.post("/auth/refresh", json={
        "refresh_token": refresh_token
    })
    assert refresh_resp.status_code == 200
    new_tokens = refresh_resp.json()
    assert "access_token" in new_tokens
    assert "refresh_token" in new_tokens

    new_access_token = new_tokens["access_token"]

    # 8. Verify the new access token can be used on protected route -> 200
    new_headers = {"Authorization": f"Bearer {new_access_token}"}
    new_me_resp = client.get("/auth/me", headers=new_headers)
    assert new_me_resp.status_code == 200
    assert new_me_resp.json()["username"] == username


def test_auth_invalid_tokens():
    """Verify signature failure and expired token rejection."""
    # 1. Invalid signature
    bad_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJib2IiLCJleHAiOjF9.invalid_signature"
    headers = {"Authorization": f"Bearer {bad_token}"}
    resp = client.get("/auth/me", headers=headers)
    assert resp.status_code == 401

    # 2. Token type mismatch (trying to use refresh token on a route requiring access token)
    # Let's register a temporary user to get a refresh token
    client.post("/auth/register", json={
        "username": "type_mismatch_user",
        "password": "password"
    })
    login_resp = client.post("/auth/token", data={
        "username": "type_mismatch_user",
        "password": "password"
    })
    refresh_token = login_resp.json()["refresh_token"]

    # Access protected route with refresh token
    headers_refresh = {"Authorization": f"Bearer {refresh_token}"}
    resp_mismatch = client.get("/auth/me", headers=headers_refresh)
    assert resp_mismatch.status_code == 401
    
    res_json = resp_mismatch.json()
    err_msg = res_json["error"]["message"] if "error" in res_json else res_json.get("detail", "")
    assert "Invalid token type" in err_msg
