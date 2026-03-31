from src.api.schemas.request import TicketSchema

RULE_NAME = "RULE_MONTH_TO_MONTH_COMPLAINT"


def check_contract_complaint(
    contract: str,
    tickets: list[TicketSchema],
) -> str | None:
    """Return rule name if Month-to-Month contract AND any complaint ticket."""
    is_monthly = contract.strip().lower() == "month-to-month"
    has_complaint = any(t.category.strip().lower() == "complaint" for t in tickets)
    return RULE_NAME if is_monthly and has_complaint else None
