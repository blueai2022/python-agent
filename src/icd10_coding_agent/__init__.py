"""ICD-10 medical coding assistant."""

from ._embeddings import EmbeddingBackend
from .cli import Settings, app, resolve_settings
from .code_aliasing import (
    Alias,
    AliasError,
    SearchableAlias,
    UnknownAliasTargetError,
    load_aliases,
    validate_aliases,
)
from .coding_agent import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_STEPS,
    AgentError,
    CaseInput,
    CodingAgent,
    FinalAnswer,
)
from .icd10_corpus import (
    CATEGORY_PREFIX_LENGTH,
    ClinicalNotes,
    Corpus,
    CorpusEntry,
    CorpusError,
    load_corpus,
)
from .llm_tool_calling import (
    BackendError,
    ChatBackend,
    Message,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolExecutionError,
    execute_tool_calls,
)
from .semantic_index_build import (
    DEFAULT_BATCH_SIZE,
    FORMAT_VERSION,
    Index,
    IndexBuildError,
    IndexEntry,
    build_index,
    build_index_entries,
    load_index,
)
from .semantic_retrieval import RetrievalError, RetrievalResult, retrieve

__all__ = [
    "CATEGORY_PREFIX_LENGTH",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_STEPS",
    "FORMAT_VERSION",
    "AgentError",
    "Alias",
    "AliasError",
    "BackendError",
    "CaseInput",
    "ChatBackend",
    "ClinicalNotes",
    "CodingAgent",
    "Corpus",
    "CorpusEntry",
    "CorpusError",
    "EmbeddingBackend",
    "FinalAnswer",
    "Index",
    "IndexBuildError",
    "IndexEntry",
    "Message",
    "RetrievalError",
    "RetrievalResult",
    "SearchableAlias",
    "Settings",
    "Tool",
    "ToolCall",
    "ToolDefinition",
    "ToolExecutionError",
    "UnknownAliasTargetError",
    "app",
    "build_index",
    "build_index_entries",
    "execute_tool_calls",
    "load_aliases",
    "load_corpus",
    "load_index",
    "resolve_settings",
    "retrieve",
    "validate_aliases",
]
