from datetime import date
from pydantic import BaseModel, Field


class TicketSchema(BaseModel):
    date: date
    category: str = Field(..., examples=["complaint", "billing", "technical"])
    text: str = ""


class PredictRiskRequest(BaseModel):
    customer_id: str
    contract: str = Field(..., examples=["Month-to-Month", "One year", "Two year"])
    monthly_charges: float = Field(..., gt=0)
    previous_monthly_charges: float = Field(..., gt=0)
    tickets: list[TicketSchema] = Field(default_factory=list)
