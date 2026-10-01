import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from ami_rag.api.routes import admin as admin_routes
from ami_rag.api.routes import metrics as metrics_routes
from ami_rag.api.routes import rag as rag_routes
from ami_rag.observability import init_telemetry
from ami_rag.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    worker_task = None
    if settings.WORKER_ENABLED:
        from ami_rag.core.factory import get_raganything
        from ami_rag.workers.ingest_worker import IngestWorker, _build_default_deps

        docs_repo, state_repo, queue, asset_store, image_worker = _build_default_deps(settings)
        worker = IngestWorker(
            rag_anything=await get_raganything(),
            docs_repo=docs_repo,
            state_repo=state_repo,
            queue=queue,
            asset_store=asset_store,
            settings=settings,
            image_worker=image_worker,
        )
        worker_task = asyncio.create_task(worker.run_forever())
    yield
    if worker_task is not None:
        worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await worker_task
    from ami_rag.core.factory import close_rag

    await close_rag()


def create_app() -> FastAPI:
    app = FastAPI(title="AMI RAG API", version="0.1.0", lifespan=lifespan)
    app.include_router(rag_routes.router, prefix="/v2/rag")
    app.include_router(admin_routes.router, prefix="/admin")
    app.include_router(metrics_routes.router)

    @app.get("/")
    async def root():
        return {"service": "ami-rag", "version": "0.1.0"}

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        return {"status": "ok"}

    return app


init_telemetry()
app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
