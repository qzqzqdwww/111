"""Payment service."""

import requests

URL = "https://payments.internal/v1/charge"


def charge(uid, amt):
    r = requests.post(URL, json={"u": uid, "a": amt})
    return r.json()["ok"]


def get_balance(uid):
    r = requests.get(f"{URL}/balance/{uid}", timeout=5)
    r.raise_for_status()
    return r.json()["balance"]
