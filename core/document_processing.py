import os
import streamlit as st
from langchain_core.documents import Document as LangchainDocument

from .config import PDF_STORAGE_PATH, CONTEXT_PDF_STORAGE_PATH
from .logger_config import get_logger

logger = get_logger(__name__)

SUPPORTED_CODE_EXTENSIONS = {".c", ".h"}


def save_uploaded_file(uploaded_file, storage_path=PDF_STORAGE_PATH):
    """Persist an uploaded C/H file to disk."""
    os.makedirs(storage_path, exist_ok=True)
    file_path = os.path.join(storage_path, uploaded_file.name)
    try:
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        logger.info(f"File '{uploaded_file.name}' saved to '{file_path}'.")
        return file_path
    except IOError as e:
        user_message = f"Failed to save uploaded file '{uploaded_file.name}'. An I/O error occurred: {e.strerror}."
        logger.error(f"{user_message} Please check permissions and disk space.")
        st.error(user_message)
        return None
    except Exception as e:
        user_message = f"An unexpected error occurred while saving '{uploaded_file.name}'."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        return None


def load_document(file_path):
    """Load a C or header file into a LangChain Document."""
    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    if file_extension not in SUPPORTED_CODE_EXTENSIONS:
        user_message = f"Unsupported file type: '{file_extension}' for file '{file_name}'. Only .c and .h are supported."
        logger.warning(user_message)
        st.error(user_message)
        return []

    try:
        logger.debug(f"Loading code file: {file_name}")
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        if not full_text.strip():
            logger.warning(f"Code file '{file_name}' is empty.")
            st.warning(f"Code file '{file_name}' appears to be empty.")
            return []
        logger.info(f"Successfully loaded code file: {file_name}")
        return [
            LangchainDocument(
                page_content=full_text,
                metadata={"source": file_path, "filename": file_name},
            )
        ]
    except UnicodeDecodeError as unicode_err:
        user_message = f"Failed to load code file '{file_name}': The file is not UTF-8 encoded."
        logger.error(f"{user_message} Details: {unicode_err}")
        st.error(user_message)
        return []
    except IOError as io_err:
        user_message = f"Failed to load code file '{file_name}': An I/O error occurred. {io_err.strerror}."
        logger.error(f"{user_message} Details: {io_err}")
        st.error(user_message)
        return []
    except Exception as e:
        user_message = f"An unexpected error occurred while attempting to load '{file_name}'."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        return []


def _chunk_code(text: str, max_chars: int = 1800, overlap: int = 300):
    """Chunk code into overlapping windows to keep prompts manageable."""
    if not text:
        return []
    if len(text) <= max_chars:
        return [text.strip()]

    chunks = []
    start = 0
    step = max_chars - overlap
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


def chunk_documents(raw_documents, storage_path=PDF_STORAGE_PATH, classify=False):
    """
    Chunk uploaded code files into manageable pieces. All chunks are treated as
    refactorable code segments; no semantic classification is required.
    """
    if not raw_documents:
        logger.warning("chunk_documents called with no raw documents.")
        st.warning("No content found in the uploaded code to chunk.")
        return [], [], []

    logger.info(f"Chunking {len(raw_documents)} code file(s).")

    all_chunks = []
    requirements_chunks = []  # reused downstream; represents code chunks
    processed_chunk_texts = set()

    try:
        for doc in raw_documents:
            chunk_texts = _chunk_code(doc.page_content)
            for chunk_text in chunk_texts:
                if not chunk_text or chunk_text in processed_chunk_texts:
                    continue
                processed_chunk_texts.add(chunk_text)
                lc_doc = LangchainDocument(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "in_memory": False},
                )
                all_chunks.append(lc_doc)
                requirements_chunks.append(lc_doc)

        logger.info(f"Code chunking complete: {len(all_chunks)} chunks created.")
        return [], requirements_chunks, all_chunks

    except Exception as e:
        logger.exception(f"An error occurred during code chunking. Details: {e}")
        st.error("An error occurred during code chunking. Check logs for details.")
        return [], [], []


def index_documents(document_chunks, vector_db=None):
    if not document_chunks:
        logger.warning("index_documents called with no chunks to index.")
        st.warning("No document chunks available to index.")
        return
    logger.info(f"Indexing {len(document_chunks)} document chunks.")
    try:
        if vector_db is None:
            vector_db = st.session_state.DOCUMENT_VECTOR_DB
        vector_db.add_documents(document_chunks)
        st.session_state.document_processed = True
        logger.info("Document chunks indexed successfully into vector store.")
    except Exception as e:
        user_message = "An error occurred while indexing document chunks."
        logger.exception(f"{user_message} Details: {e}")
        st.error(f"{user_message} Check logs for details.")
        st.session_state.document_processed = False


def re_index_documents_from_session():
    """
    Re-indexes documents from chunks stored in the session state.
    This is used to repopulate in-memory vector databases after a session is loaded.
    """
    logger.info("Attempting to re-index documents from session state.")

    # Re-index requirements (code) chunks
    if "requirements_chunks" in st.session_state and st.session_state.requirements_chunks:
        logger.info(f"Re-indexing {len(st.session_state.requirements_chunks)} code chunks.")
        index_documents(st.session_state.requirements_chunks, vector_db=st.session_state.DOCUMENT_VECTOR_DB)
    else:
        logger.info("No code chunks found in session state to re-index.")

    # Re-index standalone context chunks
    if "context_chunks" in st.session_state and st.session_state.context_chunks:
        logger.info(f"Re-indexing {len(st.session_state.context_chunks)} context chunks.")
        index_documents(st.session_state.context_chunks, vector_db=st.session_state.CONTEXT_VECTOR_DB)
        st.session_state.context_document_loaded = True
    else:
        logger.info("No standalone context chunks found in session state to re-index.")
