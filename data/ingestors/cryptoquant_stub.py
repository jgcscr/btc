"""Placeholder for CryptoQuant on-chain ingestion."""
from __future__ import annotations

CRYPTOQUANT_TICKET_ID = "CQ-2025-1213"
CURRENT_SCOPE = "daily-only access; hourly metrics pending support enablement"


def skip_cryptoquant_ingestion() -> None:
    """Log why CryptoQuant ingestion is skipped until support unlocks hourly resolution."""
    message = (
        "Skipping CryptoQuant ingestion: awaiting hourly scope approval "
        f"(ticket {CRYPTOQUANT_TICKET_ID}, scope='{CURRENT_SCOPE}')."
    )
    print(message)


__all__ = ["skip_cryptoquant_ingestion", "CRYPTOQUANT_TICKET_ID", "CURRENT_SCOPE"]
