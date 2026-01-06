import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Any

from .config import CODE_OUTPUT_PATH
from .database import (
    create_requirement_job,
    update_requirement_job,
    get_requirement_job as db_get_requirement_job,
    get_latest_requirement_job as db_get_latest_requirement_job,
    list_requirement_jobs as db_list_requirement_jobs,
)
from .generation import generate_refactored_code
from .logger_config import get_logger

logger = get_logger(__name__)
OUTPUT_ROOT = Path(CODE_OUTPUT_PATH)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class CodeRefactorJobManager:
    """Manages background code refactoring jobs."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="refactor-job")
        self._active_jobs: dict[str, Any] = {}
        logger.info("CodeRefactorJobManager initialized with %s workers.", max_workers)

    @classmethod
    def instance(cls) -> "CodeRefactorJobManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def submit_refactor_job(
        self,
        *,
        user_id: int,
        language_model,
        code_documents: List[Any],
    ) -> str:
        job_id = str(uuid.uuid4())
        metadata = {
            "code_document_count": len(code_documents or []),
        }
        create_requirement_job(job_id, user_id, status="queued", metadata=metadata)
        logger.info("Queued code refactor job %s for user %s", job_id, user_id)

        future = self._executor.submit(
            self._execute_job,
            job_id,
            user_id,
            language_model,
            code_documents or [],
        )
        self._active_jobs[job_id] = future
        future.add_done_callback(lambda _: self._active_jobs.pop(job_id, None))
        return job_id

    def _execute_job(
        self,
        job_id: str,
        user_id: int,
        language_model,
        code_documents: List[Any],
    ) -> None:
        logger.info("Starting code refactor job %s", job_id)
        update_requirement_job(job_id, status="running")
        try:
            usable_chunks = [
                chunk for chunk in code_documents if getattr(chunk, "page_content", "").strip()
            ]
            if not usable_chunks:
                raise ValueError("No usable code content found to refactor.")

            refactored_source = generate_refactored_code(language_model, usable_chunks)
            if not refactored_source or not refactored_source.strip():
                raise ValueError("Refactoring produced an empty result.")

            job_dir = OUTPUT_ROOT / str(user_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            output_path = job_dir / f"{job_id}.c"
            output_path.write_text(refactored_source, encoding="utf-8")

            metadata = {
                "code_files": [
                    getattr(chunk, "metadata", {}).get("filename")
                    or getattr(chunk, "metadata", {}).get("source")
                    for chunk in usable_chunks
                ],
                "char_count": len(refactored_source),
            }

            update_requirement_job(
                job_id,
                status="completed",
                result_path=str(output_path),
                metadata=metadata,
            )
            logger.info("Code refactor job %s completed.", job_id)
        except Exception as exc:
            logger.exception("Code refactor job %s failed: %s", job_id, exc)
            update_requirement_job(job_id, status="failed", error_message=str(exc))


# Convenience exports
def submit_code_refactor_job(**kwargs) -> str:
    return CodeRefactorJobManager.instance().submit_refactor_job(**kwargs)


def get_requirement_job(job_id: str):
    return db_get_requirement_job(job_id)


def get_latest_requirement_job(user_id: int):
    return db_get_latest_requirement_job(user_id)


def list_requirement_jobs(user_id: int, limit: int = 5):
    return db_list_requirement_jobs(user_id, limit)


def load_job_code_bytes(job_info) -> bytes | None:
    if not job_info:
        return None
    path = job_info.get("result_path")
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_bytes()


def load_job_code_text(job_info):
    if not job_info:
        return None
    path = job_info.get("result_path")
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception:
        logger.warning("Failed to read stored refactored code at %s", path)
        return None

