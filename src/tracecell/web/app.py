"""TraceCell 的 Web 应用。

界面只做一件事：把账本摊开，让人**点一个字段就看到它在原图上的那一格**。
和普通 OCR 工具的区别在于给出一条可以核对的链路，光给一段文本没用。

默认跑在回放模式（不需要 GPU），把 ``--provider minicpm`` 打开就会走真实
的 MiniCPM-OCR / Jev / MiniCPM-V 服务。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..agent.loop import VerificationLoop
from ..extractors import build_extractor
from ..reconcile import reconcile
from ..scenarios import list_scenarios
from ..export import export_csv, export_excel
from ..ledger import export_markdown, ledger_summary
from ..providers import layout as layout_mod
from ..providers.jev import HttpJevScorer
from ..providers.ocr import OpenAICompatOcr
from ..providers.replay import ReplayEscalator, ReplayParser, ReplayScorer, ReplaySession
from ..providers.vlm import OpenAICompatVlm
from ..policy import BudgetPolicy, RiskPolicy
from ..routing import render_review_checklist
from ..types import DocumentLedger

ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = ROOT / "samples"
SESSIONS_DIR = SAMPLES_DIR / "sessions"
STATIC_DIR = Path(__file__).resolve().parent / "static"


class Store:
    """内存里的账本仓库。重启即清空——这是 Demo，不引入数据库依赖。"""

    def __init__(self) -> None:
        self.ledgers: dict[str, DocumentLedger] = {}
        self.images: dict[str, bytes] = {}
        self.meta: dict[str, dict[str, Any]] = {}

    def put(self, ledger: DocumentLedger, image: bytes, meta: dict[str, Any] | None = None) -> None:
        self.ledgers[ledger.doc_id] = ledger
        self.images[ledger.doc_id] = image
        self.meta[ledger.doc_id] = meta or {}


def _load_sessions() -> dict[str, ReplaySession]:
    sessions: dict[str, ReplaySession] = {}
    if not SESSIONS_DIR.exists():
        return sessions
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            sessions[path.stem] = ReplaySession.load(path)
        except Exception:  # noqa: BLE001 - 坏文件不应该让整个服务起不来
            continue
    return sessions


def _merged_session() -> ReplaySession:
    """把 sessions 目录下的所有会话合并，样例按场景分文件存放。"""

    return ReplaySession.load_dir(SESSIONS_DIR)


def create_app(provider: str = "replay") -> FastAPI:
    app = FastAPI(title="TraceCell", version=__version__, description="可追责的文档解析 Agent")
    sessions = _load_sessions()
    merged = _merged_session()
    store = Store()

    config = {
        "provider": provider,
        "ocr_url": os.environ.get("TRACECELL_OCR_URL", "http://127.0.0.1:8000"),
        "ocr_model": os.environ.get("TRACECELL_OCR_MODEL", "openbmb/MiniCPM-OCR-1.0"),
        "jev_url": os.environ.get("TRACECELL_JEV_URL", "http://127.0.0.1:8030"),
        "vlm_url": os.environ.get("TRACECELL_VLM_URL", "http://127.0.0.1:8001"),
        "vlm_model": os.environ.get("TRACECELL_VLM_MODEL", "openbmb/MiniCPM-V-4.7-35B-A3B"),
    }

    def build_loop(doc_id: str, scenario: str) -> VerificationLoop:
        policy = BudgetPolicy()
        risk = RiskPolicy()
        if provider == "minicpm":
            parser = OpenAICompatOcr(base_url=config["ocr_url"], model=config["ocr_model"])
            scorer = HttpJevScorer(base_url=config["jev_url"])
            escalator = OpenAICompatVlm(base_url=config["vlm_url"], model=config["vlm_model"])
        else:
            session = merged
            if session is None:
                raise HTTPException(status_code=503, detail="没有可用的回放会话")
            parser = ReplayParser(session, doc_id)
            scorer = ReplayScorer(session, doc_id)
            escalator = ReplayEscalator(session, doc_id)
        return VerificationLoop(
            parser=parser,
            scorer=scorer,
            escalator=escalator,
            policy=policy,
            risk_policy=risk,
            extractor=build_extractor(scenario),
            scenario=scenario,
        )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "provider": provider}

    @app.get("/api/scenarios")
    def scenarios() -> list[dict[str, Any]]:
        return list_scenarios()

    @app.get("/api/samples")
    def samples() -> list[dict[str, Any]]:
        session = merged
        if session is None:
            return []
        out = []
        for doc_id in session.doc_ids():
            document = session.document(doc_id) or {}
            out.append(
                {
                    "doc_id": doc_id,
                    "source": document.get("source", ""),
                    "label": document.get("label", doc_id),
                    "schema": document.get("schema", "invoice"),
                    "description": document.get("description", ""),
                }
            )
        return out

    @app.get("/api/ledger/{doc_id}")
    def ledger(doc_id: str) -> JSONResponse:
        if doc_id not in store.ledgers:
            session = merged
            document = session.document(doc_id) if session else None
            if document is None:
                raise HTTPException(status_code=404, detail=f"没有文档 {doc_id}")
            image = _read_sample_image(document.get("source", ""))
            scenario = document.get("schema", "invoice")
            report = build_loop(doc_id, scenario).run(image, doc_id, document.get("source", ""))
            store.put(report.ledger, image, {"scenario": scenario, "label": document.get("label", "")})
        payload = store.ledgers[doc_id].to_dict()
        payload["summary"] = ledger_summary(store.ledgers[doc_id])
        payload["markdown"] = export_markdown(store.ledgers[doc_id])
        payload["csv"] = export_csv(store.ledgers[doc_id])
        return JSONResponse(payload)

    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...), scenario: str = "invoice") -> JSONResponse:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="空文件")
        doc_id = file.filename or "upload"
        doc_id = doc_id.rsplit(".", 1)[0]
        report = build_loop(doc_id, scenario).run(content, doc_id, file.filename or "")
        store.put(report.ledger, content, {"scenario": scenario, "label": "上传文件"})
        payload = report.ledger.to_dict()
        payload["summary"] = ledger_summary(report.ledger)
        return JSONResponse(payload)

    @app.get("/api/reconcile")
    def reconcile_all() -> JSONResponse:
        """把所有样例的账本放在一起做跨文档对账。"""

        ledgers = []
        for doc_id in merged.doc_ids():
            document = merged.document(doc_id) or {}
            try:
                report = build_loop(doc_id, document.get("schema", "invoice")).run(
                    _read_sample_image(document.get("source", "")), doc_id,
                    document.get("source", ""),
                )
            except HTTPException:
                continue
            ledgers.append(report.ledger)
            store.put(report.ledger, _read_sample_image(document.get("source", "")), {})
        result = reconcile(ledgers)
        return JSONResponse(result.to_dict())

    @app.get("/api/export/{doc_id}.xlsx")
    def export_xlsx(doc_id: str) -> Response:
        """导出 Excel 工作簿。"""

        ledger_obj = store.ledgers.get(doc_id)
        if ledger_obj is None:
            raise HTTPException(status_code=404, detail="账本尚未生成")
        return Response(
            content=export_excel(ledger_obj),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{doc_id}.xlsx"'
            },
        )

    @app.get("/api/export/{doc_id}.csv")
    def export_csv_endpoint(doc_id: str) -> Response:
        ledger_obj = store.ledgers.get(doc_id)
        if ledger_obj is None:
            raise HTTPException(status_code=404, detail="账本尚未生成")
        return Response(
            content=export_csv(ledger_obj).encode("utf-8-sig"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{doc_id}.csv"'},
        )

    @app.post("/api/feedback")
    async def submit_feedback(payload: dict[str, Any]) -> JSONResponse:
        """提交人工复核结论，回灌到账本。"""

        from ..feedback import ReviewFeedback, apply_all

        doc_id = payload.get("doc_id")
        if not doc_id:
            raise HTTPException(status_code=400, detail="缺少 doc_id")
        ledger_obj = store.ledgers.get(doc_id)
        if ledger_obj is None:
            raise HTTPException(status_code=404, detail="账本尚未生成")
        item = ReviewFeedback(
            doc_id=doc_id,
            region_id=payload.get("region_id", ""),
            verdict=payload.get("verdict", "correct"),
            corrected_text=payload.get("corrected_text", ""),
            reviewer=payload.get("reviewer", ""),
            note=payload.get("note", ""),
        )
        apply_all(ledger_obj, [item])
        return JSONResponse({"applied": True, "doc_id": doc_id})

    @app.get("/api/checklist/{doc_id}")
    def checklist(doc_id: str) -> list[dict[str, Any]]:
        ledger_obj = store.ledgers.get(doc_id)
        if ledger_obj is None:
            raise HTTPException(status_code=404, detail="账本尚未生成")
        region_map = {r.id: r for r in ledger_obj.regions}
        return render_review_checklist(region_map, ledger_obj.fields, ledger_obj.review_queue)

    @app.get("/api/image/{doc_id}")
    def image(doc_id: str) -> Response:
        content = store.images.get(doc_id)
        if content is None:
            raise HTTPException(status_code=404, detail="没有原图")
        return Response(content=content, media_type="image/png")

    @app.get("/api/crop/{doc_id}/{region_id}")
    def crop(doc_id: str, region_id: str) -> Response:
        ledger_obj = store.ledgers.get(doc_id)
        image_bytes = store.images.get(doc_id)
        if ledger_obj is None or image_bytes is None:
            raise HTTPException(status_code=404, detail="账本尚未生成")
        region = ledger_obj.region(region_id)
        if region is None:
            raise HTTPException(status_code=404, detail=f"没有区域 {region_id}")
        return Response(content=layout_mod.crop(image_bytes, region.bbox, 0.04), media_type="image/png")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app


def _read_sample_image(source: str) -> bytes:
    path = (ROOT / source).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"样例图不存在：{source}")
    return path.read_bytes()


app = create_app()
