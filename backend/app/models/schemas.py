from datetime import datetime
from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    num_chunks: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentStatus(BaseModel):
    id: str
    status: str
    num_chunks: int
    error_message: str | None

    model_config = {"from_attributes": True}
