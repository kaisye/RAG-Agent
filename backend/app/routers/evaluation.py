"""
Evaluation endpoints:
  POST /evaluation/generate-testset  — sinh testset từ PDF (background)
  GET  /evaluation/generate-testset/status
  POST /evaluation/run               — chạy RAGAS evaluate_pipeline (background)
  GET  /evaluation/run/status
  POST /evaluation/ablation          — chạy ablation study 8 configs (background)
  GET  /evaluation/ablation/status   — tiến độ (config hiện tại / 8)
  GET  /evaluation/ablation/results  — bảng JSON từ ablation_summary.csv
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
_ablation_status: dict = {
    "running": False, "done": False, "error": "",
    "current_config": 0, "total_configs": 8, "current_label": "",
    "results": [],
}


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


# ---------------------------------------------------------------------------
# POST /evaluation/ablation — chạy ablation study 8 configs
# ---------------------------------------------------------------------------

class AblationRequest(BaseModel):
    document_id: str
    testset_path: str = str(_TESTSET_PATH)
    sleep_between: float = 1.5
    resume: bool = True  # bỏ qua config đã có file kết quả


def _run_ablation_sync(document_id: str, testset_path: str, sleep_between: float, resume: bool):
    """Chạy ablation đồng bộ trong thread pool, cập nhật _ablation_status."""
    import asyncio as _asyncio
    global _ablation_status

    from scripts.run_ablation import ABLATION_CONFIGS, _save_summary_table
    from app.services.evaluation.evaluator import evaluate_pipeline as _eval

    total = len(ABLATION_CONFIGS)
    _ablation_status = {
        "running": True, "done": False, "error": "",
        "current_config": 0, "total_configs": total, "current_label": "",
        "results": [],
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    baseline_scores = None

    try:
        for i, (label, config) in enumerate(ABLATION_CONFIGS):
            _ablation_status["current_config"] = i + 1
            _ablation_status["current_label"] = label

            out_path = _RESULTS_DIR / f"config_{i}.json"

            if resume and out_path.exists():
                logger.info("[%d/%d] SKIP (cached): %s", i + 1, total, label)
                result = json.loads(out_path.read_text(encoding="utf-8"))
            else:
                logger.info("[%d/%d] Running: %s", i + 1, total, label)
                result = _asyncio.run(
                    _eval(config, document_id, testset_path, sleep_between)
                )
                out_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            all_results.append(result)
            _ablation_status["results"] = [r["scores"] for r in all_results]

            if i == 0:
                baseline_scores = result["scores"]

        _save_summary_table(all_results, baseline_scores)
        _ablation_status.update({"running": False, "done": True, "current_config": total})

    except Exception as exc:
        logger.exception("Ablation failed at config %d", _ablation_status["current_config"])
        _ablation_status.update({"running": False, "done": False, "error": str(exc)})
        raise


@router.post("/ablation", status_code=202)
async def run_ablation(req: AblationRequest, background_tasks: BackgroundTasks):
    """
    Khởi chạy ablation study — 8 configs tuần tự trong background.
    Poll GET /evaluation/ablation/status để xem tiến độ.
    Kết quả từng config lưu vào evaluation/results/config_{i}.json.
    Bảng tổng hợp xuất ra ablation_summary.csv + .md khi hoàn tất.
    """
    global _ablation_status
    if _ablation_status.get("running"):
        raise HTTPException(
            status_code=409,
            detail=f"Ablation đang chạy (config {_ablation_status['current_config']}/8).",
        )

    testset = Path(req.testset_path)
    if not testset.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Testset not found: {req.testset_path}. Run /evaluation/generate-testset first.",
        )

    background_tasks.add_task(
        asyncio.get_event_loop().run_in_executor,
        None,
        _run_ablation_sync,
        req.document_id,
        req.testset_path,
        req.sleep_between,
        req.resume,
    )

    return {
        "status": "accepted",
        "message": "Ablation study started (8 configs). Poll /evaluation/ablation/status.",
        "document_id": req.document_id,
        "total_configs": 8,
        "resume": req.resume,
    }


@router.get("/ablation/status")
async def ablation_status():
    """
    Tiến độ ablation study hiện tại.
    current_config/total_configs cho phép frontend hiển thị progress bar.
    """
    return dict(_ablation_status)


@router.get("/ablation/results")
async def ablation_results():
    """
    Bảng kết quả ablation từ ablation_summary.csv.
    Trả về list[dict] — mỗi dict là một hàng (thực nghiệm + 4 metrics + 4 delta).
    """
    from scripts.run_ablation import RESULTS_DIR as _ABL_RESULTS_DIR, load_summary_table

    if not (_ABL_RESULTS_DIR / "ablation_summary.csv").exists():
        raise HTTPException(
            status_code=404,
            detail="Chưa có kết quả ablation. Chạy POST /evaluation/ablation trước.",
        )

    rows = load_summary_table()
    return {"rows": rows, "num_configs": len(rows)}
