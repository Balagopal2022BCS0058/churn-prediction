from datetime import date, timedelta

from src.api.schemas.request import TicketSchema

RULE_NAME = "RULE_TICKET_FREQUENCY_HIGH"


def check_ticket_frequency(
    tickets: list[TicketSchema],
    window_days: int = 30,
    threshold: int = 5,
    reference_date: date | None = None,
) -> str | None:
    """Return rule name if ticket count in window exceeds threshold, else None."""
    ref = reference_date or date.today()
    cutoff = ref - timedelta(days=window_days)
    count = sum(1 for t in tickets if t.date >= cutoff)
    return RULE_NAME if count > threshold else None
