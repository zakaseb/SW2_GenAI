import streamlit as st
import pandas as pd
import json
import io
import re
from langchain_core.prompts import ChatPromptTemplate
from .config import (
    GENERAL_QA_PROMPT_TEMPLATE,
    REQUIREMENT_JSON_PROMPT_TEMPLATE,
    SUMMARIZATION_PROMPT_TEMPLATE,
    KEYWORD_EXTRACTION_PROMPT_TEMPLATE,
    CHUNK_CLASSIFICATION_PROMPT_TEMPLATE,
    CODE_REWRITE_PROMPT_TEMPLATE,
)
from .logger_config import get_logger

logger = get_logger(__name__)


def generate_answer(
    language_model,
    user_query,
    context_documents,
    conversation_history="",
    persistent_memory="",
):
    """
    Generate an answer based on the user query, context documents, and conversation history.
    """
    if not user_query or not user_query.strip():
        logger.warning("generate_answer called with empty user_query.")
        return "Your question is empty. Please type a question to get an answer."

    if not context_documents or not isinstance(context_documents, list) or len(context_documents) == 0:
        logger.warning("generate_answer called with no context documents.")
        return "I couldn't find relevant information in the document to answer your query. Please try rephrasing your question or ensure the document contains the relevant topics."

    logger.info(f"Generating answer for query: '{user_query[:50]}...'")
    try:
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        if not context_text.strip():
            logger.warning("Context text for answer generation is empty after joining docs.")
            return "The relevant sections found in the document appear to be empty. Cannot generate an answer."

        logger.debug(f"Context for prompt: {context_text[:100]}...")
        conversation_prompt = ChatPromptTemplate.from_template(GENERAL_QA_PROMPT_TEMPLATE)
        response_chain = conversation_prompt | language_model

        response = response_chain.invoke(
            {
                "user_query": user_query,
                "document_context": context_text,
                "conversation_history": conversation_history,
                "persistent_memory": persistent_memory,
            }
        )

        if not response or not response.strip():
            logger.warning("AI model returned an empty response for answer generation.")
            return "The AI model returned an empty response. Please try rephrasing your question or try again later."
        logger.info("Answer generated successfully.")
        return response
    except Exception as e:
        user_message = "I'm sorry, but I encountered an error while trying to generate a response."
        logger.exception(f"Error during answer generation: {e}")
        return f"{user_message} Please try again later or rephrase your question. (Details: {e})"


def generate_refactored_code(language_model, code_documents):
    """
    Rewrite the uploaded C/H files into a single, self-contained C source.
    """
    if not code_documents:
        logger.warning("generate_refactored_code called with no code documents.")
        return "No code was provided to refactor."

    try:
        bundle_parts = []
        for doc in code_documents:
            filename = doc.metadata.get("filename") or doc.metadata.get("source") or "code.c"
            bundle_parts.append(f"// FILE: {filename}\n{doc.page_content}")
        code_bundle = "\n\n".join(bundle_parts)
        logger.debug(f"Code bundle length for refactor: {len(code_bundle)} characters.")

        prompt = ChatPromptTemplate.from_template(CODE_REWRITE_PROMPT_TEMPLATE)
        response_chain = prompt | language_model
        response = response_chain.invoke({"code_bundle": code_bundle})

        if not response or not response.strip():
            logger.warning("AI model returned an empty response for code refactoring.")
            return "The AI model returned an empty response while refactoring the code. Please try again."

        logger.info("Refactored code generated successfully.")
        return response
    except Exception as e:
        user_message = "Failed to refactor the uploaded code."
        logger.exception(f"Error during code refactoring: {e}")
        return f"{user_message} Please try again later. (Details: {e})"


def generate_requirements_json(language_model, requirement_chunk, verification_methods_context: str = "", general_context: str = ""):
    """
    Generates a JSON object for a single requirement chunk.
    """
    logger.info("Generating requirements JSON...")
    try:
        context_text = requirement_chunk.page_content
        if not context_text.strip():
            logger.warning("generate_requirements_json called with empty chunk text.")
            return "{}"

        prompt = ChatPromptTemplate.from_template(REQUIREMENT_JSON_PROMPT_TEMPLATE)
        response_chain = prompt | language_model

        response = response_chain.invoke({
            "document_context": context_text,
            "verification_methods_context": verification_methods_context,
            "general_context": general_context,
        })

        if not response or not response.strip():
            logger.warning("AI model returned an empty response for requirements JSON generation.")
            return "{}"
        logger.info("Requirements JSON generated successfully.")
        return response
    except Exception as e:
        user_message = "I'm sorry, but I encountered an error while trying to generate the requirements JSON."
        logger.exception(f"Error during requirements JSON generation: {e}")
        return f"{{ 'error': '{user_message}', 'details': '{e}' }}"


def generate_summary(language_model, full_document_text):
    """
    Generates a summary for the given document text.
    """
    if not full_document_text or not full_document_text.strip():
        logger.warning("generate_summary called with empty document text.")
        st.warning(
            "Document content is empty or contains only whitespace. Cannot generate summary."
        )
        return None
    logger.info("Generating summary...")
    try:
        summary_prompt = ChatPromptTemplate.from_template(SUMMARIZATION_PROMPT_TEMPLATE)
        summary_chain = summary_prompt | language_model
        summary = summary_chain.invoke({"document_text": full_document_text})
        if not summary or not summary.strip():
            logger.warning("AI model returned an empty summary.")
            st.warning(
                "The AI model returned an empty summary. The document might be too short or lack clear content for summarization."
            )
            return None
        logger.info("Summary generated successfully.")
        return summary
    except Exception as e:
        user_message = "Failed to generate summary due to an AI model error."
        logger.exception(f"Error during summary generation: {e}")
        st.error(
            f"An error occurred while generating the document summary using the AI model. Details: {e}"
        )
        return f"{user_message} Please try again later. (Details: {e})"


def classify_chunk(language_model, chunk_text):
    """
    Classifies a text chunk as 'General Context' or 'Requirements'.
    """
    if not chunk_text or not chunk_text.strip():
        logger.warning("classify_chunk called with empty chunk_text.")
        return "Requirements"  # Default classification

    logger.debug(f"Classifying chunk: '{chunk_text[:50]}...'")
    try:
        classification_prompt = ChatPromptTemplate.from_template(
            CHUNK_CLASSIFICATION_PROMPT_TEMPLATE
        )
        classification_chain = classification_prompt | language_model
        response = classification_chain.invoke({"chunk_text": chunk_text})

        if not response or not response.strip():
            logger.warning("AI model returned an empty response for chunk classification.")
            return "Requirements"  # Default classification

        #The model's output might have newlines or extra spaces.
        cleaned_response = response.strip()

        if "General Context" in cleaned_response:
            logger.info("Chunk classified as General Context.")
            return "General Context"
        elif "Requirements" in cleaned_response:
            logger.info("Chunk classified as Requirements.")
            return "Requirements"
        else:
            logger.warning(
                f"Unexpected response from AI model during chunk classification: '{cleaned_response}'"
            )
            return "Requirements"  # Default to 'Requirements'

    except Exception as e:
        logger.exception(f"Error during chunk classification: {e}")
        # In case of an error, default to 'Requirements' to be safe.
        return "Requirements"


def parse_requirements_payload(requirements_json_list):
    """
    Parses and cleans a list of JSON strings representing requirements.
    Returns a list of requirement dictionaries.
    """
    all_requirements = []
    for json_str in requirements_json_list:
        # Clean the string: remove markdown and other non-JSON artifacts
        # This regex looks for content between ```json and ``` or just `{` and `}` or `[` and `]`
        match = re.search(r"```json\s*([\s\S]*?)\s*```|([\s\S]*)", json_str)
        if match:
            cleaned_str = match.group(1) if match.group(1) is not None else match.group(2)
            cleaned_str = cleaned_str.strip()

            try:
                # Try to parse the cleaned string
                data = json.loads(cleaned_str)
                if isinstance(data, list):
                    all_requirements.extend(data)
                elif isinstance(data, dict):
                    all_requirements.append(data)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from string: {cleaned_str}")
                continue # Skip this string if it's not valid JSON

    return all_requirements


def generate_excel_file(requirements_json_list):
    """
    Parses a list of JSON strings, cleans them, and generates an Excel file in memory.
    """
    all_requirements = parse_requirements_payload(requirements_json_list)

    if not all_requirements:
        return None

    # Define the columns based on the JSON schema to ensure order and handle missing keys
    columns = [
        "Name",
        "Description",
        "VerificationMethod",
        "Tags",
        "RequirementType",
        "DocumentRequirementID"
    ]

    # Create a DataFrame
    df = pd.DataFrame(all_requirements)

    # Ensure all columns are present, fill missing ones with empty strings
    for col in columns:
        if col not in df.columns:
            df[col] = ''

    # Reorder columns to match the desired schema and select only them
    df = df[columns]

    # Convert list-like columns (e.g., Tags) to a string representation
    if 'Tags' in df.columns:
        df['Tags'] = df['Tags'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Create an in-memory Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Requirements')

    processed_data = output.getvalue()
    return processed_data
