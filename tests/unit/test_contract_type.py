import pytest
from datetime import date, timedelta
from src.api.schemas.request import TicketSchema
from src.engine.rules.contract_type import check_contract_complaint, RULE_NAME


def make_ticket(category: str) -> TicketSchema:
    return TicketSchema(date=date.today() - timedelta(days=1), category=category, text="")


def test_triggers_monthly_with_complaint():
    tickets = [make_ticket("complaint")]
    assert check_contract_complaint("Month-to-Month", tickets) == RULE_NAME


def test_no_trigger_yearly_with_complaint():
    tickets = [make_ticket("complaint")]
    assert check_contract_complaint("One year", tickets) is None


def test_no_trigger_monthly_no_complaint():
    tickets = [make_ticket("billing"), make_ticket("technical")]
    assert check_contract_complaint("Month-to-Month", tickets) is None


def test_no_trigger_empty_tickets():
    assert check_contract_complaint("Month-to-Month", []) is None


def test_case_insensitive_contract():
    tickets = [make_ticket("complaint")]
    assert check_contract_complaint("month-to-month", tickets) == RULE_NAME


def test_case_insensitive_category():
    tickets = [make_ticket("Complaint")]
    assert check_contract_complaint("Month-to-Month", tickets) == RULE_NAME
