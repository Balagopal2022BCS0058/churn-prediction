from src.api.schemas.request import TicketSchema

RULE_NAME = "RULE_CHARGE_INCREASE_MEDIUM"


def check_charge_increase(
    monthly_charges: float,
    previous_monthly_charges: float,
    tickets: list[TicketSchema],
    ticket_threshold: int = 3,
) -> str | None:
    """Return rule name if charges increased AND ticket count >= threshold."""
    charges_increased = monthly_charges > previous_monthly_charges
    enough_tickets = len(tickets) >= ticket_threshold
    return RULE_NAME if charges_increased and enough_tickets else None
