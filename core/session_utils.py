import streamlit as st

# Define keys that should be persisted across sessions for a user.
# We explicitly avoid saving things that are not pickle-able or are temporary,
# like file uploaders, download triggers, or the models themselves.
PERSISTENT_KEYS = [
    "messages",
    "memory",
    "raw_documents",
    "uploaded_filenames",
    "processed_files_info",
    "document_processed",
    "bm25_index",
    "bm25_corpus_chunks",
    "general_context_chunks",
    "requirements_chunks",
    "processed_context_file_info",
    "context_document_loaded",
    "context_chunks",
    "refactored_code_text",
    "refactored_code_bytes",
]

def package_session_for_storage():
    """Creates a dictionary of session state values that should be persisted."""
    packaged_state = {}
    for key in PERSISTENT_KEYS:
        if key in st.session_state:
            packaged_state[key] = st.session_state[key]
    return packaged_state

def unpack_session_from_storage(loaded_state):
    """Populates the current session_state with values from storage."""
    if loaded_state:
        for key, value in loaded_state.items():
            st.session_state[key] = value
