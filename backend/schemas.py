# backend/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# Repository Ingestion / Status
class AnalyzeRepositoryRequest(BaseModel):
    github_url: str = Field(..., description="Full GitHub repository URL")
    llm_provider: str = Field("gemini", description="gemini | ollama")
    top_k: int = Field(8, ge=1, le=50, description="Number of chunks to retrieve per query")

class RepositoryStatus(BaseModel):
    repository_id: str
    name: str
    owner: str
    description: Optional[str] = None
    primary_language: Optional[str] = None
    languages: List[str] = []
    file_count: int = 0
    chunk_count: int = 0
    indexing_status: str = "not_started"  # not_started | indexing | ready | failed
    error_message: Optional[str] = None

# Chat / Q&A
class AskRequest(BaseModel):
    repository_id: str
    question: str
    top_k: int = Field(8, ge=1, le=50)

class SourceRefResponse(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceRefResponse] = []
    chunks_retrieved: int = 0
    model_name: str = ""

# Architecture
class ArchitectureResponse(BaseModel):
    dependency_graph: Dict[str, List[str]] = {}
    mermaid_diagram: str = ""
    summary: str = ""

# Dependencies
class DependencyResponse(BaseModel):
    name: str
    version_spec: str
    ecosystem: str
    source_file: str
    dependency_type: str = "direct"
    extras: List[str] = []

# Bugs
class BugFindingResponse(BaseModel):
    tool: str
    severity: str
    file_path: str
    line: int
    column: int
    code: str
    issue: str
    explanation: str
    recommendation: str
    confidence: str

# Security
class SecurityFindingResponse(BaseModel):
    file_path: str
    line: int
    category: str
    pattern_name: str
    masked_value: str
    recommendation: str
    confidence: str

# GitHub Issues
class AnalyzeIssueRequest(BaseModel):
    repository_id: str
    issue_number: int

class IssueAnalysisResponse(BaseModel):
    issue_number: int
    title: str
    state: str
    likely_affected_files: List[SourceRefResponse] = []
    probable_root_cause: str = ""
    related_components: List[str] = []
    confidence: str = "LOW"
    full_analysis: str = ""
    labels: List[str] = []

# Pull Requests
class AnalyzePRRequest(BaseModel):
    repository_id: str
    pr_number: int

class ReviewFindingResponse(BaseModel):
    category: str
    description: str
    confidence: str
    file_path: Optional[str] = None
    line: Optional[int] = None

class PRReviewResponse(BaseModel):
    pr_number: int
    title: str
    state: str
    changed_files: List[str] = []
    findings: List[ReviewFindingResponse] = []
    overall_assessment: str = ""
    diff_truncated: bool = False

# Error Response
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
