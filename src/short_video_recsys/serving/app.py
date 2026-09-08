"""FastAPI application for the two-stage recommendation pipeline."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request

from short_video_recsys.config import load_yaml
from short_video_recsys.serving.pipeline import RecommendationPipeline


def create_app(pipeline: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = pipeline is None
        if owned:
            config_path = Path(
                os.environ.get("RECSYS_SERVING_CONFIG", "configs/serving.yaml")
            )
            app.state.pipeline = RecommendationPipeline(load_yaml(config_path))
        else:
            app.state.pipeline = pipeline
        yield
        if owned:
            app.state.pipeline.close()

    app = FastAPI(
        title="KuaiRand Two-Stage Recommender",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        active = request.app.state.pipeline
        return {"status": "ok", "feature_store": active.backend_name}

    @app.get("/recommend/{user_id}")
    def recommend(
        request: Request,
        user_id: int,
        top_k: int = Query(default=10, ge=1, le=100),
        tab: int = Query(default=1, ge=0),
    ) -> dict[str, Any]:
        result = request.app.state.pipeline.recommend(user_id, top_k, tab)
        if result is None:
            raise HTTPException(status_code=404, detail="Unknown user_id")
        return result

    return app


app = create_app()


def serve(config: dict[str, Any]) -> None:
    import uvicorn

    uvicorn.run(
        create_app(RecommendationPipeline(config)),
        host=os.environ.get("RECSYS_API_HOST", str(config["api"]["host"])),
        port=int(os.environ.get("RECSYS_API_PORT", config["api"]["port"])),
        log_level="info",
        access_log=bool(config["api"].get("access_log", False)),
    )
