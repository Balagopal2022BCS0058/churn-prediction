from datetime import date, timedelta

from src.api.schemas.request import TicketSchema
from src.engine.rules.charge_increase import RULE_NAME, check_charge_increase


def make_ticket(category: str = "billing") -> TicketSchema:
    return TicketSchema(date=date.today() - timedelta(days=1), category=category, text="")


def test_triggers_when_charge_up_and_enough_tickets():
    tickets = [make_ticket() for _ in range(3)]
    assert check_charge_increase(90.0, 70.0, tickets) == RULE_NAME


def test_no_trigger_when_charges_same():
    tickets = [make_ticket() for _ in range(3)]
    assert check_charge_increase(70.0, 70.0, tickets) is None


def test_no_trigger_when_charges_decreased():
    tickets = [make_ticket() for _ in range(3)]
    assert check_charge_increase(60.0, 70.0, tickets) is None


def test_no_trigger_when_too_few_tickets():
    tickets = [make_ticket(), make_ticket()]  # only 2
    assert check_charge_increase(90.0, 70.0, tickets) is None


def test_no_trigger_with_no_tickets():
    assert check_charge_increase(90.0, 70.0, []) is None


def test_custom_ticket_threshold():
    tickets = [make_ticket() for _ in range(2)]
    assert check_charge_increase(90.0, 70.0, tickets, ticket_threshold=2) == RULE_NAME
