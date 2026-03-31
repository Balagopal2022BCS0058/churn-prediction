import structlog
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from src.config import settings
from src.api.routes import health, predict
from src.api.middleware.logging import LoggingMiddleware
from src.api.middleware.metrics import MetricsMiddleware


def configure_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    structlog.get_logger().info("startup", engine=settings.engine_type.value)
    yield
    structlog.get_logger().info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Churn Prediction API",
        description="Telecom customer churn risk prediction service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)

    app.include_router(health.router)
    app.include_router(predict.router)

    # Mount Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=settings.app_host, port=settings.app_port, reload=True)
