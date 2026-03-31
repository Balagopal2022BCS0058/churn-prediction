from src.api.schemas.request import PredictRiskRequest
from src.api.schemas.response import PredictRiskResponse, RiskLevel
from src.engine.base import RiskEngine
from src.engine.rules.ticket_frequency import check_ticket_frequency
from src.engine.rules.charge_increase import check_charge_increase
from src.engine.rules.contract_type import check_contract_complaint


class RuleBasedEngine(RiskEngine):
    async def evaluate(self, request: PredictRiskRequest) -> PredictRiskResponse:
        triggered: list[str] = []

        freq_rule = check_ticket_frequency(request.tickets)
        charge_rule = check_charge_increase(
            request.monthly_charges,
            request.previous_monthly_charges,
            request.tickets,
        )
        contract_rule = check_contract_complaint(request.contract, request.tickets)

        if freq_rule:
            triggered.append(freq_rule)
        if charge_rule:
            triggered.append(charge_rule)
        if contract_rule:
            triggered.append(contract_rule)

        # Priority: HIGH > MEDIUM > LOW
        high_rules = {"RULE_TICKET_FREQUENCY_HIGH", "RULE_MONTH_TO_MONTH_COMPLAINT"}
        medium_rules = {"RULE_CHARGE_INCREASE_MEDIUM"}

        triggered_set = set(triggered)
        if triggered_set & high_rules:
            risk_level = RiskLevel.HIGH
        elif triggered_set & medium_rules:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return PredictRiskResponse(
            customer_id=request.customer_id,
            risk_level=risk_level,
            triggered_rules=triggered,
            details={
                "ticket_count": len(request.tickets),
                "monthly_charges": request.monthly_charges,
                "contract": request.contract,
            },
        )
