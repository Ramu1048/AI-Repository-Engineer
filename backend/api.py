# backend/api.py
import logging
import traceback
from dotenv import load_dotenv
load_dotenv()  # Load .env before any downstream modules read environment variables

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from typing import List, Dict

# Downstream imports
import github_service
import ingestion
import retrieval
import rag
import analyzers
import github_analysis

from exceptions import (
    IngestionError,
    InvalidGitHubURLError,
    RepoNotFoundError,
    PrivateRepoError,
    RateLimitError,
    CloneError,
    CloneTimeoutError,
    EmptyRepositoryError,
    FileDecodeError,
)
from retrieval.exceptions import (
    RepositoryNotIndexedError,
    EmptyCollectionError,
    EmbeddingProviderConfigError,
    EmbeddingProviderUnreachableError,
    VectorStoreWriteError,
    VectorStoreQueryError,
)

from backend.schemas import (
    AnalyzeRepositoryRequest,
    RepositoryStatus,
    AskRequest,
    AskResponse,
    SourceRefResponse,
    ArchitectureResponse,
    DependencyResponse,
    BugFindingResponse,
    SecurityFindingResponse,
    AnalyzeIssueRequest,
    IssueAnalysisResponse,
    AnalyzePRRequest,
    PRReviewResponse,
    ReviewFindingResponse,
    ErrorResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Repository Engineer API",
    description="Backend API for AI Repository Analysis, RAG, and bug/security assessment",
    version="1.0.0"
)

# In-memory store for repository ingestion state
_REPOSITORIES: Dict[str, RepositoryStatus] = {}

# Exception helper for cleaner user messages
def handle_exception(exc: Exception):
    logger.error(f"Error caught: {exc}")
    logger.error(traceback.format_exc())
    
    if isinstance(exc, InvalidGitHubURLError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    elif isinstance(exc, RepoNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    elif isinstance(exc, PrivateRepoError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))
    elif isinstance(exc, RateLimitError):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))
    elif isinstance(exc, CloneTimeoutError):
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc))
    elif isinstance(exc, CloneError):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    elif isinstance(exc, EmptyRepositoryError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    elif isinstance(exc, FileDecodeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    elif isinstance(exc, RepositoryNotIndexedError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    elif isinstance(exc, EmptyCollectionError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    elif isinstance(exc, EmbeddingProviderConfigError):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    elif isinstance(exc, EmbeddingProviderUnreachableError):
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc))
    elif isinstance(exc, VectorStoreWriteError):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    elif isinstance(exc, VectorStoreQueryError):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    elif isinstance(exc, HTTPException):
        raise exc
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {exc}"
        )


def _get_repository_id(owner: str, name: str) -> str:
    target_prefix = f"{owner.lower()}_{name.lower()}_"
    for repo_key in _REPOSITORIES:
        if repo_key.lower().startswith(target_prefix):
            return repo_key
    # Fallback to legacy format or mock tests
    fallback_id = f"{owner.lower()}/{name.lower()}"
    for repo_key in _REPOSITORIES:
        if repo_key.lower() == fallback_id:
            return repo_key
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Repository '{owner}/{name}' has not been analyzed yet. Please submit to /api/analyze first."
    )

def _run_indexing_background(repository_id: str, repo_url, github_meta, gh_client):
    """Performs the actual long-running clone, chunk, embed, and index process in a background thread."""
    try:
        # Step 4: Shallow-clone
        clone_path = github_service.clone_repository(
            owner=repo_url.owner,
            repository=repo_url.repository,
            repository_id=repository_id,
            default_branch=github_meta["default_branch"],
            token=gh_client.token
        )
        
        # Step 5: Load codebase files
        loaded_files = ingestion.load_repository_files(repository_id)
        
        # Count total and skipped files for metadata builder
        import os
        total_files = sum(len(files) for _, _, files in os.walk(clone_path))
        skipped_files = total_files - len(loaded_files)
        
        # Step 6: Build final metadata
        meta = ingestion.build_metadata(
            repository_id=repository_id,
            owner=repo_url.owner,
            repository=repo_url.repository,
            github_meta=github_meta,
            loaded_files=loaded_files,
            file_count_total=total_files,
            file_count_skipped=skipped_files,
        )
        
        # Step 7: Chunk files
        dict_chunks = ingestion.chunk_repository(repository_id, loaded_files)
        chunks = [
            retrieval.Chunk(
                repository_id=dc["repository_id"],
                file_path=dc["file_path"],
                language=dc["language"] or "unknown",
                symbol_type=dc["symbol_type"] or "doc_section",
                symbol_name=dc["symbol_name"],
                class_name=dc["class_name"],
                start_line=dc["start_line"],
                end_line=dc["end_line"],
                content=dc["content"]
            )
            for dc in dict_chunks
        ]
        
        # Step 8: Store chunks in retrieval index (with real embedding generation & fallback)
        try:
            from embeddings import get_embedding_provider
            provider = get_embedding_provider()
            embeddings_list = provider.embed([c.content for c in chunks])
        except Exception as e:
            logger.warning(f"Could not generate real embeddings ({e}), falling back to mock embeddings.")
            embeddings_list = [[0.1] * 384 for _ in chunks]

        retrieval.store_chunks(repository_id, chunks, embeddings_list)
        
        # Step 9: Finalize status as ready
        status_obj = _REPOSITORIES[repository_id]
        status_obj.file_count = meta.file_count_indexed
        status_obj.languages = meta.primary_languages
        status_obj.primary_language = meta.primary_languages[0] if meta.primary_languages else "unknown"
        status_obj.chunk_count = len(chunks)
        status_obj.indexing_status = "ready"
        
        logger.info(f"Successfully indexed repo {repository_id} in background with {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Background indexing failed for repo {repository_id}: {e}\n{traceback.format_exc()}")
        if repository_id in _REPOSITORIES:
            _REPOSITORIES[repository_id].indexing_status = "failed"
            _REPOSITORIES[repository_id].description = f"Indexing failed: {str(e)[:200]}"


@app.post("/api/analyze", response_model=RepositoryStatus, responses={500: {"model": ErrorResponse}})
def analyze_repository(request: AnalyzeRepositoryRequest, background_tasks: BackgroundTasks):
    """
    Ingests, chunks, and indexes a GitHub repository asynchronously.
    """
    url = request.github_url
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid GitHub URL. Must start with http:// or https://"
        )

    try:
        # Step 1: Parse URL
        repo_url = github_service.parse_github_url(url)
        
        # Step 2: Fetch metadata
        gh_client = github_service.GitHubClient()
        github_meta = gh_client.fetch_repo_metadata(repo_url.owner, repo_url.repository)
        
        # Step 3: Generate repository ID
        repository_id = github_service.generate_repository_id(
            repo_url.owner, repo_url.repository, github_meta["latest_commit_sha"]
        )
        
        # Check if already ready or indexing
        if repository_id in _REPOSITORIES:
            status_obj = _REPOSITORIES[repository_id]
            if status_obj.indexing_status in ("indexing", "ready"):
                logger.info(f"Repository {repository_id} status is already {status_obj.indexing_status}. Returning existing status.")
                return status_obj

        # Set status to indexing in memory
        status_obj = RepositoryStatus(
            repository_id=repository_id,
            name=repo_url.repository,
            owner=repo_url.owner,
            description=github_meta.get("description", ""),
            primary_language=github_meta.get("language", "unknown"),
            languages=[github_meta.get("language")] if github_meta.get("language") else [],
            file_count=0,
            chunk_count=0,
            indexing_status="indexing"
        )
        _REPOSITORIES[repository_id] = status_obj
        
        # Start background indexing task
        background_tasks.add_task(
            _run_indexing_background,
            repository_id,
            repo_url,
            github_meta,
            gh_client
        )
        
        logger.info(f"Successfully queued background indexing for repo {repository_id}.")
        return status_obj

    except Exception as exc:
        handle_exception(exc)

@app.get("/api/repository/{owner}/{name}", response_model=RepositoryStatus)
def get_repository(owner: str, name: str):
    """
    Gets status and details of a previously indexed repository.
    """
    repository_id = _get_repository_id(owner, name)
    # Update chunk count dynamically
    status_obj = _REPOSITORIES[repository_id]
    status_obj.chunk_count = retrieval.chunk_count(repository_id)
    return status_obj

@app.post("/api/chat", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Queries the RAG chain with a question about a repository.
    """
    repo_id = request.repository_id
    if repo_id not in _REPOSITORIES or _REPOSITORIES[repo_id].indexing_status != "ready":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Repository not indexed or not ready. Please analyze the repository first."
        )
        
    try:
        rag_answer = rag.ask(repo_id, request.question)
        
        # Convert downstream sources to API responses
        sources_response = [
            SourceRefResponse(
                file_path=src.file_path,
                start_line=src.start_line,
                end_line=src.end_line,
                symbol_name=src.symbol_name
            ) for src in rag_answer.sources
        ]
        
        return AskResponse(
            answer=rag_answer.answer,
            sources=sources_response,
            chunks_retrieved=len(rag_answer.sources),
            model_name="Gemini 3.5 Flash (Mocked)"
        )
    except Exception as exc:
        handle_exception(exc)

@app.get("/api/repository/{owner}/{name}/architecture", response_model=ArchitectureResponse)
def get_architecture(owner: str, name: str):
    """
    Performs architectural analysis.
    """
    repo_id = _get_repository_id(owner, name)
         
    try:
        report = analyzers.analyze_architecture(repo_id)
        return ArchitectureResponse(
            dependency_graph=report.dependency_graph,
            mermaid_diagram=report.mermaid_diagram,
            summary=report.summary
        )
    except Exception as exc:
        handle_exception(exc)

@app.get("/api/repository/{owner}/{name}/dependencies", response_model=List[DependencyResponse])
def get_dependencies(owner: str, name: str):
    """
    Performs dependency analysis.
    """
    repo_id = _get_repository_id(owner, name)
         
    try:
        deps = analyzers.analyze_dependencies(repo_id)
        return [
            DependencyResponse(
                name=dep.name,
                version_spec=dep.version,
                ecosystem=dep.ecosystem,
                source_file=dep.file_source,
                dependency_type="direct" if dep.is_direct else "dev",
                extras=[dep.outdated_status] if dep.outdated_status else []
            ) for dep in deps
        ]
    except Exception as exc:
        handle_exception(exc)

@app.get("/api/repository/{owner}/{name}/bugs", response_model=List[BugFindingResponse])
def get_bugs(owner: str, name: str):
    """
    Performs bug analysis.
    """
    repo_id = _get_repository_id(owner, name)
         
    try:
        findings = analyzers.analyze_bugs(repo_id)
        return [
            BugFindingResponse(
                tool="ast",
                severity=f.severity,
                file_path=f.file_path,
                line=f.line_number,
                column=0,
                code=f.issue_code,
                issue=f.issue_title,
                explanation=f.explanation,
                recommendation=f.recommendation,
                confidence=f.confidence
            ) for f in findings
        ]
    except Exception as exc:
        handle_exception(exc)

@app.get("/api/repository/{owner}/{name}/security", response_model=List[SecurityFindingResponse])
def get_security(owner: str, name: str):
    """
    Performs static security analysis.
    """
    repo_id = _get_repository_id(owner, name)
         
    try:
        findings = analyzers.analyze_security(repo_id)
        return [
            SecurityFindingResponse(
                file_path=f.file_path,
                line=f.line_number,
                category=f.category,
                pattern_name=f.category,
                masked_value=f.masked_value, # Must be masked
                recommendation=f.recommendation,
                confidence=f.confidence
            ) for f in findings
        ]
    except Exception as exc:
        handle_exception(exc)

@app.post("/api/repository/{owner}/{name}/issues/{issue_number}", response_model=IssueAnalysisResponse)
def analyze_repo_issue(owner: str, name: str, issue_number: int):
    """
    Analyses a GitHub issue relative to the indexed repository code.
    """
    repo_id = _get_repository_id(owner, name)
         
    try:
        analysis = github_analysis.analyze_issue(repo_id, issue_number)
        
        sources_response = [
            SourceRefResponse(
                file_path=filepath,
                start_line=1,
                end_line=1,
                symbol_name=None
            ) for filepath in analysis.likely_affected_files
        ]
        
        return IssueAnalysisResponse(
            issue_number=analysis.issue_number,
            title=analysis.title,
            state=analysis.status,
            likely_affected_files=sources_response,
            probable_root_cause=analysis.root_cause_hypothesis,
            related_components=analysis.related_components,
            confidence=analysis.confidence,
            full_analysis=f"{analysis.summary}\n\nSuggested Fix: {analysis.suggested_fix}",
            labels=[]
        )
    except Exception as exc:
        handle_exception(exc)

@app.post("/api/repository/{owner}/{name}/pulls/{pr_number}", response_model=PRReviewResponse)
def analyze_repo_pr(owner: str, name: str, pr_number: int):
    """
    Analyses a GitHub pull request and yields a detailed review.
    """
    repo_id = _get_repository_id(owner, name)
         
    try:
        review = github_analysis.analyze_pr(repo_id, pr_number)
        
        findings_response = []
        for bug in review.potential_bugs:
            findings_response.append(
                ReviewFindingResponse(
                    category="potential_bug",
                    description=bug,
                    confidence="likely"
                )
            )
        for conc in review.security_concerns:
            findings_response.append(
                ReviewFindingResponse(
                    category="security_concern",
                    description=conc,
                    confidence="likely"
                )
            )
        for tst in review.missing_tests:
            findings_response.append(
                ReviewFindingResponse(
                    category="missing_tests",
                    description=tst,
                    confidence="medium"
                )
            )
        for breaking in review.breaking_changes:
            findings_response.append(
                ReviewFindingResponse(
                    category="breaking_change",
                    description=breaking,
                    confidence="high"
                )
            )
        
        return PRReviewResponse(
            pr_number=review.pr_number,
            title=review.title,
            state="open",
            changed_files=[],
            findings=findings_response,
            overall_assessment=f"{review.summary}\n\nDocumentation Impact: {review.documentation_impact}\n\nRecommendations:\n" + "\n".join([f"- {r}" for r in review.recommendations]),
            diff_truncated=False
        )
    except Exception as exc:
        handle_exception(exc)
