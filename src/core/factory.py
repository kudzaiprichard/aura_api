from fastapi import FastAPI
from src.configs import application
from src.core.lifespan import lifespan
from src.core.middleware import register_middleware
from src.core.rate_limit import register_rate_limiter
from src.shared.exceptions.error_handlers import register_error_handlers
from src.app.controllers import (
    analysis_router,
    auth_router,
    benchmark_router,
    dashboard_router,
    drift_router,
    escalation_router,
    extension_auth_router,
    extension_email_router,
    extension_install_router,
    model_router,
    review_router,
    system_router,
    training_buffer_router,
    training_run_router,
    user_router,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title=application.name,
        version=application.version,
        debug=application.debug,
        lifespan=lifespan,
    )

    register_rate_limiter(app)
    register_middleware(app)
    register_error_handlers(app)
    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    app.include_router(system_router, tags=["System"])
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Auth"])
    # Extension auth endpoints live under the same /api/v1/auth base so the
    # router paths (/extension/register, etc.) resolve per BACKEND_CONTRACT §4.
    app.include_router(
        extension_auth_router,
        prefix="/api/v1/auth",
        tags=["Extension"],
    )
    # Extension email-analyse surface lives under its own /api/v1/emails
    # base — distinct from the dashboard's /api/v1/analysis/predict route
    # so the two surfaces never share a prefix or response shape.
    app.include_router(
        extension_email_router,
        prefix="/api/v1/emails",
        tags=["Extension"],
    )
    # Admin install management — dashboard-facing, gated by require_admin
    # via the router's dependencies (§13). Distinct prefix from the
    # extension's own auth/analyse routes so the admin surface and the
    # extension surface never collide on a path.
    app.include_router(
        extension_install_router,
        prefix="/api/v1/extension/installs",
        tags=["Extension Admin"],
    )
    app.include_router(user_router, prefix="/api/v1/users", tags=["Users"])
    app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])
    app.include_router(review_router, prefix="/api/v1/review/queue", tags=["Review"])
    app.include_router(
        escalation_router,
        prefix="/api/v1/review/escalations",
        tags=["Review"],
    )
    app.include_router(drift_router, prefix="/api/v1/drift", tags=["Drift"])
    app.include_router(
        training_buffer_router,
        prefix="/api/v1/training/buffer",
        tags=["Training"],
    )
    app.include_router(
        training_run_router,
        prefix="/api/v1/training/runs",
        tags=["Training"],
    )
    app.include_router(model_router, prefix="/api/v1/models", tags=["Models"])
    app.include_router(
        benchmark_router,
        prefix="/api/v1/benchmarks",
        tags=["Benchmarks"],
    )
    app.include_router(
        dashboard_router,
        prefix="/api/v1/dashboards",
        tags=["Dashboards"],
    )
