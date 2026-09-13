"""Account normalization and the non-negotiable M5A demo gate."""

from __future__ import annotations

import hashlib
from datetime import datetime
from decimal import Decimal

from .contracts import AccountObservation

CONTROLLER_VERSION = "M5A-1"
LIVE_REFUSAL_MESSAGE = "MR LAB M5A REFUSES LIVE ACCOUNT"


class LiveAccountRefused(RuntimeError):
    pass


def _text(value: object) -> str:
    return str(value)


def runtime_identity(broker_name: str, account_number: object) -> str:
    material = f"{CONTROLLER_VERSION}\0{broker_name}\0{account_number}\0DEMO"
    return "m5a" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def normalize_account(account: object, observed_at: datetime) -> AccountObservation:
    if bool(account.IsLive):
        raise LiveAccountRefused(LIVE_REFUSAL_MESSAGE)
    broker = _text(account.BrokerName)
    number = account.Number
    free_margin = getattr(account, "FreeMargin", None)
    asset = getattr(account, "Asset", None)
    currency = getattr(asset, "Name", None)
    if currency is None:
        currency = getattr(account, "Currency", asset or "UNKNOWN")
    return AccountObservation(
        observed_at=observed_at,
        broker_name=broker,
        account_type=_text(account.AccountType),
        currency=_text(currency),
        is_live=False,
        balance=Decimal(str(account.Balance)),
        equity=Decimal(str(account.Equity)),
        free_margin=None if free_margin is None else Decimal(str(free_margin)),
        runtime_id=runtime_identity(broker, number),
    )
