from __future__ import annotations
import os
from pathlib import Path
import json
from urllib.parse import urlparse, urlunparse


# Global Application Constants
MAX_HISTORY_TURNS = (
    3  # Number of recent user/assistant turn pairs to include in history
)
K_SEMANTIC = 20 # Number of results for semantic search
K_BM25 = 20 # Number of results for BM25 search
K_RRF_PARAM = 60  # Constant for Reciprocal Rank Fusion (RRF)
TOP_K_FOR_RERANKER = 40 #Number of docs from hybrid search to pass to reranker
FINAL_TOP_N_FOR_CONTEXT = 15  # Number of docs reranker should return for LLM context

# Prompt Templates
GENERAL_QA_PROMPT_TEMPLATE = """
You are a senior C/C++ engineer. Use the provided code context, conversation history, and persistent memory to answer the current query about the uploaded codebase.
- Keep answers concise, actionable, and grounded in the supplied code.
- If the information is absent, clearly say you do not know and avoid inventing details.

Persistent Memory (if any):
{persistent_memory}

Conversation History (if any):
{conversation_history}

Code Context:
{document_context}

Current Query: {user_query}
Answer:
"""

REQUIREMENT_JSON_PROMPT_TEMPLATE = """
You are an expert system engineer specialized in requirement extraction. Your task is to analyze the provided text chunk and extract all the requirements it contains.
For each requirement found in the text chunk, you must generate a single JSON object that follows the schema below.
If a chunk contains multiple requirements, generate a list of JSON objects.
If the chunk contains no requirements, return an empty list [].
Your response MUST be only the JSON data (a single object or a list of objects) and nothing else. Do not include any prefixes, suffixes, or explanations.

JSON Schema:
{{
  "Name": "string",
  "Description": "string",
  "VerificationMethod": "string (e.g., Analysis, Inspection, Demonstration, Test)",
  "Tags": "list of strings",
  "RequirementType": "string (e.g., Functional, Constraint)",
  "DocumentRequirementID": "string"
}}
whereby:

Name: The name of the requirement
Description: The requirement description
Verification Method: This should be 'Test', 'Inspection' or 'Analysis'. The verification 
   method of each requirement can be identified in table 4-4, where the requirement ID can 
   be mapped to a verification method
Tags: In the case where there is a TBD or TBC in the description, add the tag 'TBD' in 
   this column for a requirement.
Requirement Type: Please classify the requirement as one of the following - 'Functional', 
   'Interface' or 'Constraint'
Document Requirement ID: This should be the unique randomised alphanumerical requirement ID of each requirement.

When extracting the requirements, please ensure that the following are adhered to: 
1. Requirements are verifiable.
2. Requirements are autonomous.
3. Requirements are unambiguous.
4. Requirements are concise.

In the case where requirements are not meeting the above, please mark them in the 'Tags' 
column as 'Updated' by adding the word 'Updated' in the Tags column (note: if there is a 
TBD in the column, then separate them with a comma). Please update the description to adhere 
to these requirements for the requirements quality. Feel free to split requirements into 
multiple requirements if so required.

Here is an example of a desired JSON object:
{{
  "Name": "Dual-Mode HMI",
  "Description": "The system shall support dual-mode hmi as per mission profile and design objectives.",
  "VerificationMethod": "Analysis",
  "Tags": ["TBD"],
  "RequirementType": "Functional",
  "DocumentRequirementID": "#177"
}}

Now, analyze the following text chunk and extract the requirements.

Text Chunk:
{document_context}

Additional reference materials you MUST use while generating/validating requirements:

- Verification Methods (fixed offline reference; use this first to determine 'VerificationMethod'):
{verification_methods_context}

- General Context (persistent knowledge; use this to clarify terms, interfaces, and constraints):
{general_context}

JSON Output:
"""

SUMMARIZATION_PROMPT_TEMPLATE = """
You are an expert research assistant. Provide a concise summary of the following document.
Focus on the main points and key takeaways. The summary should be approximately 3-5 sentences long.

Document:
{document_text}

Summary:
"""

KEYWORD_EXTRACTION_PROMPT_TEMPLATE = """
You are an expert research assistant. Analyze the following document and extract the top 5-10 most relevant keywords or key phrases.
Present them as a comma-separated list.

Document:
{document_text}

Keywords:
"""

CHUNK_CLASSIFICATION_PROMPT_TEMPLATE = """
You are an expert document analyst. Classify the following text chunk into one of two categories:
1.  **General Context**: Portions of a document that provide broad, high-level information. This includes introductions, overviews, and background information that help a reader understand the overall context of the document, but do not contain specific, detailed requirements.
2.  **Requirements**: Portions of a document that contain specific, detailed, and actionable requirements, specifications, or instructions. These are the granular details of a project, system, or process. A requirement should be a statement that can be verified or tested.

I will provide below an example of both classes to get a better understanding of the required task. 

An example of the "General Context" class is:

"Introduction:
This document specifies the technical system requirements for the All-Terrain Military Hybrid Vehicle (ATMHV), translating user needs (as defined in the URS) into detailed, implementable, and verifiable system requirements. The A-Spec outlines functional, performance, interface, design, safety, and verification criteria to guide system development, integration, and testing.
System Overview:
The ATMHV is an 8-passenger hybrid-powered vehicle designed for military operations across challenging terrains including deserts, forests, swamps, and snow. The vehicle provides off-road mobility, tactical communication capabilities, protection against threats, and logistics support. It must operate reliably across extreme environmental conditions.
"

An example of the "Requirements" class is:

"       SS_TO_SR-SOI_SYS_REQ-1
Maximum Speed Limitation #1
The system shall support maximum speed limitation #1 as per mission profile and design objectives. 
Verification Method(s): Analysis
       SS_TO_SR-SOI_SYS_REQ-2
Environmental Operating Range #2
The system shall support environmental operating range #2 as per mission profile and design objectives. 
Verification Method(s): Inspection
       SS_TO_SR-SOI_SYS_REQ-3
Minimum Range Capability #3
The system shall support minimum range capability #3 as per mission profile and design objectives. 
Verification Method(s): Demonstration
       SS_TO_SR-SOI_SYS_REQ-4
Dual-Mode Powertrain #4
The system shall support dual-mode powertrain #4 as per mission profile and design objectives. 
Verification Method(s): Test
       SS_TO_SR-SOI_SYS_REQ-5
Dynamic Power Mode Switching #5
The system shall support dynamic power mode switching #5 as per mission profile and design objectives. 
Verification Method(s): Test
"
Analyze the text chunk provided below and determine which category it belongs to.
Your response should be a single word: either "General Context" or "Requirements".

Text Chunk:
{chunk_text}

Classification:
"""

CODE_REWRITE_PROMPT_TEMPLATE = """
You are a senior C engineer. Rewrite the provided C/H sources into a single, self-contained C implementation that:
- Preserves all observable behavior and interfaces.
- Renames functions, variables, macros, and types to new, non-trivial names.
- Reorders logic, restructures control flow, and rewrites comments so the result is not textually traceable to the inputs.
- Removes unused code and clarifies unclear sections while keeping functionality intact.
- Emits clean, compilable ANSI C (C11 or later) without Markdown, fences, or explanations.

Source bundle (each file is delimited):
{code_bundle}

Return only the rewritten C source file content. Do not include headers, Markdown fences, or narrative text.
"""

# ========== Hard defaults (no env required) ==========
# Models
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text"  
OLLAMA_LLM_NAME             = "codellama:13b"
RERANKER_MODEL_NAME         = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Storage
BASE_STORE                  = Path("./document_store")
CODE_STORAGE_PATH           = str(BASE_STORE / "code_inputs")
CONTEXT_CODE_STORAGE_PATH   = str(BASE_STORE / "context_code")
PDF_STORAGE_PATH            = CODE_STORAGE_PATH  # Backward compatibility
CONTEXT_PDF_STORAGE_PATH    = CONTEXT_CODE_STORAGE_PATH
MEMORY_FILE_PATH            = str(BASE_STORE / "memory" / "context.json")
REQUIREMENTS_OUTPUT_PATH    = str(BASE_STORE / "generated_requirements")  # legacy name
CODE_OUTPUT_PATH            = str(BASE_STORE / "refactored_code")

# Endpoints (local single-device defaults)
# OLLAMA_BASE_URL is what the API wrapper uses to reach Ollama.
# RAG_API_BASE is what the Streamlit app uses to reach your wrapper.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
RAG_API_BASE    = "http://127.0.0.1:8000"

# Wrapper switch + (optional) auth
USE_API_WRAPPER = True
RAG_API_KEY     = ""  # set token here if you add auth to api.py

_local_path = os.path.join(os.path.dirname(__file__), "config.local.json")
if os.path.exists(_local_path):
    try:
        _ov = json.load(open(_local_path, "r"))
        for k, v in _ov.items():
            if k in globals():
                globals()[k] = v
    except Exception:
        pass

# ========== Utilities shared by API + App ==========
def _normalize_url(url: str, default_port: int) -> str:
    """Force IPv4 for 'localhost' and ensure a port is present."""
    u = urlparse(url)
    host = u.hostname or "127.0.0.1"
    port = u.port or default_port
    if host == "localhost":
        host = "127.0.0.1"
    return urlunparse((u.scheme or "http", f"{host}:{port}", "", "", "", ""))

# Canonical, normalized URLs to use everywhere else
OLLAMA_URL = _normalize_url(OLLAMA_BASE_URL, 11434)  # your “ollama_url”
API_URL    = _normalize_url(RAG_API_BASE,    8000)

# Postman / debug proxy (set only when you want to capture traffic)
# export POSTMAN_PROXY=http://127.0.0.1:5559
POSTMAN_PROXY = os.getenv("POSTMAN_PROXY") 

# Ensure directories exist
for p in (
    PDF_STORAGE_PATH,
    CONTEXT_PDF_STORAGE_PATH,
    Path(MEMORY_FILE_PATH).parent,
    REQUIREMENTS_OUTPUT_PATH,
    CODE_OUTPUT_PATH,
):
    Path(p).mkdir(parents=True, exist_ok=True)
