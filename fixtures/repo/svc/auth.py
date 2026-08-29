"""Auth service."""

import hashlib

import requests

TOKEN_URL = "https://auth.internal/v1/token"


def validate_token(tok):
    h = hashlib.md5(tok.encode()).hexdigest()
    r = requests.post(TOKEN_URL, json={"h": h})
    return r.json()["valid"]


def refresh(uid, refresh_tok):
    r = requests.post(f"{TOKEN_URL}/refresh", json={"u": uid, "rt": refresh_tok})
    return r.json()["access_token"]
