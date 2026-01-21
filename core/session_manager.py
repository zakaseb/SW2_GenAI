import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from .model_loader import get_embedding_model
from .logger_config import get_logger
from .database import delete_session
import os
from .document_processing import load_document, chunk_documents, index_documents
from .config import CONTEXT_PDF_STORAGE_PATH
from pathlib import Path
import shutil

logger = get_logger(__name__)

def purge_persistent_memory():
    """
    Purge persistent chat state for the current user and reset ONLY the context
    document (and its vector base). General document VB(s) are preserved.
    """
    user_id = st.session_state.get("user_id")
    if not user_id:
        logger.warning("Attempted to purge memory without a user_id in session.")
        st.error("Cannot purge memory: User not identified.")
        return

    # 1) Delete the session from the persistent database (chat history, etc.)
    try:
        delete_session(user_id)
        logger.info(f"Successfully purged persistent session data for user_id: {user_id}.")
    except Exception as e:
        logger.exception(f"Failed to purge session for user_id: {user_id}. Error: {e}")
        st.error("Failed to purge persistent memory from the database.")
        return

    # 2) Reset in-memory chat state ONLY (do not touch general doc VB)
    st.session_state.memory = []
    st.session_state["chat_history"] = []

    # 3) Reset CONTEXT doc state & CONTEXT VB (keep GENERAL VB intact)
    try:
        embedding_model = st.session_state.get("EMBEDDING_MODEL") or get_embedding_model()
        st.session_state.CONTEXT_VECTOR_DB = InMemoryVectorStore(embedding_model)
    except Exception as e:
        logger.warning(f"Could not reset CONTEXT_VECTOR_DB cleanly: {e}")

    st.session_state.context_document_loaded = False
    st.session_state.processed_context_file_info = None
    st.session_state.context_chunks = []

    # IMPORTANT: prevent auto-bootstrap right after purge
    st.session_state["allow_global_context"] = False
    st.session_state["did_context_bootstrap"] = True   # block the auto-load block
    st.session_state.pop("global_ctx_indexed_mtime", None)

    # Clear the context uploader widget state (prevents ghost file)
    st.session_state.pop("context_file_uploader", None)

    # 4) Delete the global fixed context docs on disk (once)
    try:
        ctx_dir = Path(CONTEXT_PDF_STORAGE_PATH)
        if ctx_dir.exists():
            for p in ctx_dir.iterdir():
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                    else:
                        shutil.rmtree(p)
                    logger.info(f"Removed context file/folder: {p}")
                except Exception as e:
                    logger.exception(f"Failed to remove {p}: {e}")
    except Exception as e:
        logger.exception(f"Failed while cleaning context directory: {e}")

    logger.info(f"In-memory session state reset (chat cleared, context purged) for user_id: {user_id}.")

def initialize_session_state():
    """
    Initializes the session state variables if they don't exist.
    """
    # Create a second vector DB for verification methods that you won't purge
    if "PERSISTENT_VECTOR_DB" not in st.session_state:
        st.session_state.PERSISTENT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    if "CONTEXT_VECTOR_DB" not in st.session_state:
        st.session_state.CONTEXT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    if "GENERAL_VECTOR_DB" not in st.session_state:
        st.session_state.GENERAL_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    if "DOCUMENT_VECTOR_DB" not in st.session_state:
        st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "context_document_loaded" not in st.session_state:
        st.session_state.context_document_loaded = False
    if "uploaded_file_key" not in st.session_state:
        st.session_state.uploaded_file_key = 0
    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = []
    if "raw_documents" not in st.session_state:
        st.session_state.raw_documents = []
    if "processed_files_info" not in st.session_state:
        st.session_state.processed_files_info = {}
    if "processed_context_file_info" not in st.session_state:
        st.session_state.processed_context_file_info = None
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
    if "bm25_corpus_chunks" not in st.session_state:
        st.session_state.bm25_corpus_chunks = []
    if "refactored_code_text" not in st.session_state:
        st.session_state.refactored_code_text = None
    if "refactored_code_bytes" not in st.session_state:
        st.session_state.refactored_code_bytes = None


def reset_document_states(clear_chat=True):
    """
    Resets all document-related session state variables.
    Optionally clears chat history.
    """
    # Re-initialize vector stores
    st.session_state.DOCUMENT_VECTOR_DB = InMemoryVectorStore(get_embedding_model())

    # Reset file processing info
    st.session_state.document_processed = False
    st.session_state.processed_files_info = {}
    st.session_state.uploaded_filenames = []
    st.session_state.raw_documents = []

    # Reset generated content
    st.session_state.refactored_code_text = None
    st.session_state.refactored_code_bytes = None

    # Reset chunks and search indices
    st.session_state.general_context_chunks = []
    st.session_state.requirements_chunks = []
    st.session_state.bm25_index = None
    st.session_state.bm25_corpus_chunks = []

    # Optionally clear chat history
    if clear_chat:
        st.session_state.messages = []

    logger.info("Document states reset.")


def reset_file_uploader():
    """Increments the key for the file uploader to reset it."""
    st.session_state.uploaded_file_key += 1
