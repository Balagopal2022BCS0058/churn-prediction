from datetime import date, timedelta

from src.api.schemas.request import TicketSchema
from src.engine.rules.ticket_frequency import RULE_NAME, check_ticket_frequency


def make_ticket(days_ago: int, category: str = "general") -> TicketSchema:
    return TicketSchema(date=date.today() - timedelta(days=days_ago), category=category, text="")


def test_triggers_when_above_threshold():
    tickets = [make_ticket(i) for i in range(1, 7)]  # 6 tickets
    assert check_ticket_frequency(tickets) == RULE_NAME


def test_does_not_trigger_at_threshold():
    tickets = [make_ticket(i) for i in range(1, 6)]  # exactly 5
    assert check_ticket_frequency(tickets) is None


def test_does_not_trigger_below_threshold():
    tickets = [make_ticket(i) for i in range(1, 4)]  # 3 tickets
    assert check_ticket_frequency(tickets) is None


def test_empty_tickets():
    assert check_ticket_frequency([]) is None


def test_excludes_tickets_outside_window():
    # 4 tickets within 30 days, 3 older tickets
    tickets = [make_ticket(i) for i in range(1, 5)]  # 4 within window
    tickets += [make_ticket(35), make_ticket(40), make_ticket(50)]  # outside
    assert check_ticket_frequency(tickets) is None


def test_custom_threshold():
    tickets = [make_ticket(i) for i in range(1, 4)]  # 3 tickets
    assert check_ticket_frequency(tickets, threshold=2) == RULE_NAME


def test_boundary_date_excluded():
    # ticket exactly on cutoff day (31 days ago) is excluded from 30d window
    ref = date.today()
    tickets = [make_ticket(31)] + [make_ticket(i) for i in range(1, 7)]
    result = check_ticket_frequency(tickets, window_days=30, threshold=5, reference_date=ref)
    assert result == RULE_NAME  # 6 within window
