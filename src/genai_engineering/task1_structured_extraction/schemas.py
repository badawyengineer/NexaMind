from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Party(StrictModel):
    name: str | None = None
    address: str | None = None
    email: str | None = None
    phone: str | None = None
    tax_id: str | None = None


class LineItem(StrictModel):
    description: str
    quantity: Decimal | None = Field(default=None, ge=0)
    unit_price: Decimal | None = Field(default=None, ge=0)
    amount: Decimal | None = Field(default=None, ge=0)


class Invoice(StrictModel):
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    vendor: Party | None = None
    customer: Party | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal | None = Field(default=None, ge=0)
    tax: Decimal | None = Field(default=None, ge=0)
    total: Decimal | None = Field(default=None, ge=0)
    currency: str | None = None
    notes: list[str] = Field(default_factory=list)


class Contact(StrictModel):
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    links: list[str] = Field(default_factory=list)


class Education(StrictModel):
    institution: str
    degree: str | None = None
    field: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class WorkExperience(StrictModel):
    organization: str
    title: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: list[str] = Field(default_factory=list)


class Resume(StrictModel):
    full_name: str | None = None
    contact: Contact = Field(default_factory=Contact)
    professional_summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
