from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.logging_config import configure_logging
from app.routes import health, models, predict, trips


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    logger = configure_logging()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info(
            "Starting %s v%s | trip_data_dir=%s | exists=%s",
            app_settings.app_name,
            app_settings.app_version,
            app_settings.trip_data_dir,
            app_settings.trip_data_dir.is_dir(),
        )
        yield
        logger.info("Shutting down %s", app_settings.app_name)

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=app_settings.app_description,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: app_settings

    # CORS configuration for React + Vite dev server and two-laptop LAN deployment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_origin_regex=app_settings.cors_origin_regex,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Structured CMD/stdout request logging (mandatory for live evaluator verification)
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        client_host = request.client.host if request.client else "unknown"
        logger.info(
            "%s %s | status=%d | duration=%.2fms | client=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            client_host,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ):
        messages = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", []) if part != "body")
            msg = err.get("msg", "Invalid input")
            messages.append(f"{loc}: {msg}" if loc else msg)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "detail": "; ".join(messages) or "Invalid request payload.",
                "code": "validation_error",
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception):
        logger.exception("Unhandled backend error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred.",
                "code": "internal_server_error",
            },
        )

    # Primary /api-prefixed routes matching frontend/src/services/pathsnapApi.js
    for route_module in (health, trips, models, predict):
        app.include_router(route_module.router, prefix="/api")

    # Root-level compatibility aliases (/health, /trips, /models, /models/comparison, /predict)
    for route_module in (health, trips, models, predict):
        app.include_router(route_module.router, include_in_schema=False)

    return app


app = create_app()
