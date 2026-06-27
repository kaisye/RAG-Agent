"""
Evaluation endpoints:
  POST /evaluation/generate-testset  — sinh testset từ PDF (background)
  GET  /evaluation/generate-testset/status
  POST /evaluation/run               — chạy RAGAS evaluate_pipeline (background)
  GET  /evaluation/run/status
"""
import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.config import PipelineConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])

_TESTSET_PATH = Path("evaluation/testset.json")
_SAMPLE_PDF_DIR = "evaluation/sample_docs"
_MD_DIR = "storage/markdown"
_RESULTS_DIR = Path("evaluation/results")

# Track trạng thái job đơn giản (in-memory, reset khi restart)
_job_status: dict = {"running": False, "done": False, "count": 0, "error": ""}
_eval_status: dict = {"running": False, "done": False, "result": None, "error": ""}


from pydantic import BaseModel


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


# ---------------------------------------------------------------------------
# POST /evaluation/run — chạy RAGAS evaluate_pipeline
# ---------------------------------------------------------------------------

class RunEvalRequest(BaseModel):
    document_id: str
    config: PipelineConfig = PipelineConfig()
    testset_path: str = str(_TESTSET_PATH)
    sleep_between: float = 1.5


def _run_evaluation(document_id: str, config: PipelineConfig, testset_path: str, sleep_between: float):
    """Chạy đồng bộ trong thread pool."""
    import asyncio as _asyncio
    global _eval_status
    try:
        _eval_status = {"running": True, "done": False, "result": None, "error": ""}

        from app.services.evaluation.evaluator import evaluate_pipeline as _eval
        result = _asyncio.run(_eval(config, document_id, testset_path, sleep_between))

        _eval_status = {"running": False, "done": True, "result": result, "error": ""}
        logger.info("Evaluation complete: %s", result.get("scores"))
    except Exception as exc:
        logger.exception("Evaluation failed")
        _eval_status = {"running": False, "done": False, "result": None, "error": str(exc)}
        raise


@router.post("/run", status_code=202)
async def run_evaluation(req: RunEvalRequest, background_tasks: BackgroundTasks):
    """
    Chạy RAGAS evaluate_pipeline cho một PipelineConfig.
    Kết quả được lưu vào evaluation/results/{label}.json.
    Poll GET /evaluation/run/status để lấy kết quả.
    """
    global _eval_status
    if _eval_status.get("running"):
        raise HTTPException(status_code=409, detail="Evaluation already running.")

    testset = Path(req.testset_path)
    if not testset.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Testset not found: {req.testset_path}. Run /evaluation/generate-testset first.",
        )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    background_tasks.add_task(
        asyncio.get_event_loop().run_in_executor,
        None,
        _run_evaluation,
        req.document_id,
        req.config,
        req.testset_path,
        req.sleep_between,
    )

    return {
        "status": "accepted",
        "message": "Evaluation started. Poll /evaluation/run/status for progress.",
        "document_id": req.document_id,
        "config": req.config.model_dump(),
    }


@router.get("/run/status")
async def eval_status():
    """Trả về trạng thái và kết quả của job evaluation hiện tại."""
    status = dict(_eval_status)
    # Nếu done, liệt kê thêm tất cả kết quả đã lưu
    if status.get("done") and _RESULTS_DIR.exists():
        saved = [str(p) for p in sorted(_RESULTS_DIR.glob("*.json"))]
        status["saved_results"] = saved
    return status
