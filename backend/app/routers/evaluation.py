"""
POST /evaluation/generate-testset
Chạy TestsetGenerator như background task, lưu kết quả vào evaluation/testset.json.
"""
import asyncio
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

_TESTSET_PATH = Path("evaluation/testset.json")
_SAMPLE_PDF_DIR = "evaluation/sample_docs"
_MD_DIR = "storage/markdown"

# Track trạng thái job đơn giản (in-memory, reset khi restart)
_job_status: dict = {"running": False, "done": False, "count": 0, "error": ""}


class GenerateTestsetRequest(BaseModel):
    pdf_dir: str = _SAMPLE_PDF_DIR
    md_dir: str = _MD_DIR
    output_path: str = str(_TESTSET_PATH)
    size: int = 50
    skip_convert: bool = False


def _run_generation(pdf_dir: str, md_dir: str, output_path: str, size: int, skip_convert: bool):
    """Chạy đồng bộ trong thread pool (gọi từ background task)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from scripts.generate_testset import step1_convert_pdfs, step2_load_docs, step3_generate

    global _job_status
    try:
        _job_status = {"running": True, "done": False, "count": 0, "error": ""}

        if not skip_convert:
            step1_convert_pdfs(pdf_dir, md_dir)

        docs = step2_load_docs(md_dir)
        count = step3_generate(docs, output_path, size=size)

        _job_status = {"running": False, "done": True, "count": count, "error": ""}
    except Exception as exc:
        _job_status = {"running": False, "done": False, "count": 0, "error": str(exc)}
        raise


@router.post("/generate-testset", status_code=202)
async def generate_testset(req: GenerateTestsetRequest, background_tasks: BackgroundTasks):
    """
    Khởi chạy TestsetGenerator trong background.
    Poll GET /evaluation/generate-testset/status để kiểm tra kết quả.
    """
    global _job_status
    if _job_status.get("running"):
        raise HTTPException(status_code=409, detail="Testset generation already running.")

    Path(req.md_dir).mkdir(parents=True, exist_ok=True)
    Path(req.output_path).parent.mkdir(parents=True, exist_ok=True)

    background_tasks.add_task(
        asyncio.get_event_loop().run_in_executor,
        None,
        _run_generation,
        req.pdf_dir,
        req.md_dir,
        req.output_path,
        req.size,
        req.skip_convert,
    )

    return {
        "status": "accepted",
        "message": f"Testset generation started ({req.size} samples). Poll /evaluation/generate-testset/status.",
        "output_path": req.output_path,
    }


@router.get("/generate-testset/status")
async def testset_status():
    """Trả về trạng thái job generate-testset hiện tại."""
    status = dict(_job_status)
    if status.get("done"):
        status["testset_path"] = str(_TESTSET_PATH)
    return status
