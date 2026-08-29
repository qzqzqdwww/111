"""Billing service."""

import requests

REFUND_URL = "https://payments.internal/v1/refund"


def refund(uid, amt, reason):
    r = requests.post(REFUND_URL, json={"u": uid, "a": amt, "why": reason})
    return r.json()["ok"]


def list_invoices(uid, limit=50):
    r = requests.get(f"{REFUND_URL}/invoices/{uid}", params={"n": limit}, timeout=5)
    r.raise_for_status()
    return r.json()["invoices"]
