from pydantic import BaseModel, Field


class ChatMessages(BaseModel):
    role: str
    content: str


class RagFilters(BaseModel):
    organization_unit_id: str | None = None
    document_type: str | None = None
    modality: list[str] | None = None


class RAGRequest(BaseModel):
    messages: list[ChatMessages] | None = Field(default=None)
    top_k: int | None = Field(default=None, ge=1, le=200)
    version: int = Field(default=1, ge=1, le=2)
    mode: str = Field(default="mix")
    include_references: bool = True
    include_kg: bool = False
    filters: RagFilters | None = None


class RetrievedDoc(BaseModel):
    text: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class RetrievedDocV2(RetrievedDoc):
    reference_id: str | None = None
    doc: dict | None = None
    modality: str | None = None
    artifact_url: str | None = None
    table_body: str | None = None
    page: int | None = None
    caption: str | None = None
    entities: list[dict] | None = None


class RetrievalResponse(BaseModel):
    query: str
    documents: list[RetrievedDoc]


class RetrievalResponseV2(RetrievalResponse):
    documents: list[RetrievedDocV2] = Field(default_factory=list)
    mode: str | None = None
    references: list[dict] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)
