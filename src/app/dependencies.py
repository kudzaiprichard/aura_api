import asyncio
import logging
from datetime import datetime, timezone

from fastapi import Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.sse import SSEBroker
from src.shared.database import get_db
from src.shared.exceptions import AuthenticationException, AuthorizationException
from src.shared.inference import AutoReviewer, DriftMonitor, PhishingDetector
from src.app.helpers.auto_review_cache import AutoReviewCache
from src.app.helpers.install_token_provider import hash_install_token
from src.shared.responses import ErrorDetail
from src.app.models.extension_install import ExtensionInstall
from src.app.models.enums import ExtensionInstallStatus, Role
from src.app.models.user import User
from src.app.repositories.extension_install_repository import (
    ExtensionInstallRepository,
)
from src.app.repositories.extension_token_repository import (
    ExtensionTokenRepository,
)
from src.app.repositories.user_repository import UserRepository
from src.app.repositories.token_repository import TokenRepository
from src.app.repositories.auto_review_invocation_repository import (
    AutoReviewInvocationRepository,
)
from src.app.repositories.benchmark_dataset_repository import (
    BenchmarkDatasetRepository,
)
from src.app.repositories.benchmark_dataset_row_repository import (
    BenchmarkDatasetRowRepository,
)
from src.app.repositories.drift_event_repository import DriftEventRepository
from src.app.repositories.model_benchmark_repository import (
    ModelBenchmarkRepository,
)
from src.app.repositories.model_benchmark_version_result_repository import (
    ModelBenchmarkVersionResultRepository,
)
from src.app.repositories.model_activation_repository import (
    ModelActivationRepository,
)
from src.app.repositories.model_threshold_history_repository import (
    ModelThresholdHistoryRepository,
)
from src.app.repositories.prediction_event_repository import (
    PredictionEventRepository,
)
from src.app.repositories.review_disagreement_repository import (
    ReviewDisagreementRepository,
)
from src.app.repositories.review_escalation_repository import (
    ReviewEscalationRepository,
)
from src.app.repositories.review_item_repository import ReviewItemRepository
from src.app.repositories.training_buffer_repository import (
    TrainingBufferRepository,
)
from src.app.repositories.training_run_repository import TrainingRunRepository
from src.app.services.auth_service import AuthService
from src.app.services.benchmark_service import BenchmarkService
from src.app.services.extension_auth_service import ExtensionAuthService
from src.app.services.extension_email_service import ExtensionEmailService
from src.app.services.extension_install_admin_service import (
    ExtensionInstallAdminService,
)
from src.app.services.dashboard_service import DashboardService
from src.app.services.drift_service import DriftService
from src.app.services.escalation_service import EscalationService
from src.app.services.user_management_service import UserManagementService
from src.app.services.inference_status_service import InferenceStatusService
from src.app.services.model_admin_service import ModelAdminService
from src.app.services.prediction_service import PredictionService
from src.app.services.review_service import ReviewService
from src.app.services.training_buffer_service import TrainingBufferService
from src.app.services.training_service import TrainingService
from src.configs import inference as inference_config

bearer_scheme = HTTPBearer()
# Separate scheme for the extension surface so a missing/empty header maps to
# the contract's 401 AUTH_FAILED envelope (HTTPBearer's default 403 is wrong
# for this path) and does not interfere with the dashboard's bearer scheme.
extension_bearer_scheme = HTTPBearer(auto_error=False)

_extension_logger = logging.getLogger("src.app.extension")


# ── Service factories ──

def get_auth_service(session: AsyncSession = Depends(get_db)) -> AuthService:
    return AuthService(UserRepository(session), TokenRepository(session))


def get_extension_auth_service(
    session: AsyncSession = Depends(get_db),
) -> ExtensionAuthService:
    return ExtensionAuthService(
        ExtensionInstallRepository(session),
        ExtensionTokenRepository(session),
    )


def get_extension_install_admin_service(
    session: AsyncSession = Depends(get_db),
) -> ExtensionInstallAdminService:
    return ExtensionInstallAdminService(
        session=session,
        install_repository=ExtensionInstallRepository(session),
        token_repository=ExtensionTokenRepository(session),
        prediction_event_repository=PredictionEventRepository(session),
    )


def get_user_management_service(session: AsyncSession = Depends(get_db)) -> UserManagementService:
    return UserManagementService(UserRepository(session), TokenRepository(session))


def get_review_service(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> ReviewService:
    return ReviewService(
        session=session,
        review_item_repository=ReviewItemRepository(session),
        review_escalation_repository=ReviewEscalationRepository(session),
        training_buffer_repository=TrainingBufferRepository(session),
        prediction_event_repository=PredictionEventRepository(session),
        user_repository=UserRepository(session),
        drift_monitor=request.app.state.drift_monitor,
        drift_event_repository=DriftEventRepository(session),
        auto_review_invocation_repository=AutoReviewInvocationRepository(session),
        auto_reviewer=request.app.state.auto_reviewer,
        auto_review_cache=getattr(
            request.app.state, "auto_review_cache", None
        ),
        detector=request.app.state.detector,
    )


def get_escalation_service(
    session: AsyncSession = Depends(get_db),
    review_service: ReviewService = Depends(get_review_service),
) -> EscalationService:
    return EscalationService(
        review_item_repository=ReviewItemRepository(session),
        review_escalation_repository=ReviewEscalationRepository(session),
        review_service=review_service,
        review_disagreement_repository=ReviewDisagreementRepository(session),
    )


def get_prediction_service(
    request: Request,
    session: AsyncSession = Depends(get_db),
    review_service: ReviewService = Depends(get_review_service),
) -> PredictionService:
    return PredictionService(
        prediction_repository=PredictionEventRepository(session),
        detector=request.app.state.detector,
        review_service=review_service,
        drift_event_repository=DriftEventRepository(session),
        app_state=request.app.state,
    )


def get_extension_email_service(
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> ExtensionEmailService:
    # Delegates to PredictionService so extension-sourced analyses land in
    # `prediction_events` (requester_id=NULL, request_id tagged with
    # `ext:{install_id}:...`) and REVIEW-zone verdicts are enqueued onto
    # the analyst review queue through the same path as the dashboard.
    return ExtensionEmailService(prediction_service=prediction_service)


def get_drift_service(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> DriftService:
    return DriftService(
        session=session,
        drift_event_repository=DriftEventRepository(session),
        drift_monitor=request.app.state.drift_monitor,
    )


def get_training_buffer_service(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> TrainingBufferService:
    return TrainingBufferService(
        buffer_repository=TrainingBufferRepository(session),
        detector=request.app.state.detector,
    )


def get_training_service(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> TrainingService:
    return TrainingService(
        session=session,
        training_run_repository=TrainingRunRepository(session),
        training_buffer_repository=TrainingBufferRepository(session),
        sse_broker=request.app.state.sse_broker,
        background_jobs=request.app.state.background_jobs,
        training_lock=request.app.state.training_lock,
        app_state=request.app.state,
    )


def get_benchmark_service(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> BenchmarkService:
    return BenchmarkService(
        session=session,
        dataset_repository=BenchmarkDatasetRepository(session),
        dataset_row_repository=BenchmarkDatasetRowRepository(session),
        benchmark_repository=ModelBenchmarkRepository(session),
        result_repository=ModelBenchmarkVersionResultRepository(session),
        sse_broker=request.app.state.sse_broker,
        background_jobs=request.app.state.background_jobs,
    )


def get_dashboard_service(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> DashboardService:
    buffer_repo = TrainingBufferRepository(session)
    return DashboardService(
        prediction_event_repository=PredictionEventRepository(session),
        review_item_repository=ReviewItemRepository(session),
        auto_review_invocation_repository=AutoReviewInvocationRepository(session),
        training_buffer_repository=buffer_repo,
        training_buffer_service=TrainingBufferService(
            buffer_repository=buffer_repo,
        ),
        training_run_repository=TrainingRunRepository(session),
        model_activation_repository=ModelActivationRepository(session),
        token_repository=TokenRepository(session),
        user_repository=UserRepository(session),
        detector=request.app.state.detector,
        drift_monitor=request.app.state.drift_monitor,
        models_dir=inference_config.models_dir,
    )


# ── Inference singletons (constructed in lifespan; see src/core/lifespan.py) ──

def get_detector(request: Request) -> PhishingDetector | None:
    return request.app.state.detector


def get_drift_monitor(request: Request) -> DriftMonitor:
    return request.app.state.drift_monitor


def get_auto_reviewer(request: Request) -> AutoReviewer | None:
    return request.app.state.auto_reviewer


def get_auto_review_cache(request: Request) -> AutoReviewCache | None:
    return getattr(request.app.state, "auto_review_cache", None)


def get_detector_lock(request: Request) -> asyncio.Lock:
    return request.app.state.detector_lock


def get_training_lock(request: Request) -> asyncio.Lock:
    return request.app.state.training_lock


def get_sse_broker(request: Request) -> SSEBroker:
    return request.app.state.sse_broker


def get_inference_status_service(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> InferenceStatusService:
    return InferenceStatusService(
        detector=request.app.state.detector,
        drift_monitor=request.app.state.drift_monitor,
        auto_reviewer=request.app.state.auto_reviewer,
        training_run_repository=TrainingRunRepository(session),
    )


def get_model_admin_service(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> ModelAdminService:
    return ModelAdminService(
        session=session,
        activation_repository=ModelActivationRepository(session),
        threshold_history_repository=ModelThresholdHistoryRepository(session),
        prediction_event_repository=PredictionEventRepository(session),
        detector_lock=request.app.state.detector_lock,
        drift_monitor=request.app.state.drift_monitor,
        app_state=request.app.state,
    )


# ── Auth dependencies ──

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    return await auth_service.get_current_user(credentials.credentials)


def require_role(*allowed_roles: Role):
    async def _guard(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            role_names = ", ".join(r.value for r in allowed_roles)
            raise AuthorizationException(
                message="You don't have permission to perform this action",
                error_detail=ErrorDetail(
                    title="Access Denied",
                    code="INSUFFICIENT_ROLE",
                    status=403,
                    details=[f"Required role(s): {role_names}"],
                ),
            )
        return current_user
    return _guard


# Convenience aliases
require_admin = require_role(Role.ADMIN)
require_authenticated = require_role(Role.ADMIN, Role.IT_ANALYST)


# ── Extension install auth (Chrome extension surface) ──

def _auth_failed(detail: str) -> AuthenticationException:
    return AuthenticationException(
        message="Authentication failed",
        error_detail=ErrorDetail(
            title="Authentication failed",
            code="AUTH_FAILED",
            status=401,
            details=[detail],
        ),
    )


def _not_whitelisted() -> AuthorizationException:
    return AuthorizationException(
        message="Your account is not authorised for AURA",
        error_detail=ErrorDetail(
            title="Your account is not authorised for AURA",
            code="NOT_WHITELISTED",
            status=403,
            details=["Install is blacklisted"],
        ),
    )


async def _resolve_install_from_token(
    raw_token: str | None,
    session: AsyncSession,
) -> ExtensionInstall | None:
    """Validate an install bearer token and return its install record.

    Returns `None` when no token is supplied (callers that want optional auth
    treat that as anonymous). Raises 401/403 for present-but-invalid tokens
    so optional-auth endpoints still reject malformed credentials cleanly.
    """
    if not raw_token:
        return None

    token_repo = ExtensionTokenRepository(session)
    token_row = await token_repo.get_by_hash_with_install(
        hash_install_token(raw_token)
    )
    if token_row is None or not token_row.is_valid:
        raise _auth_failed("Install token missing, expired, or revoked")

    install = token_row.install
    if install is None:
        # Defensive: the FK is non-nullable, but a delete race could in
        # principle leave the join empty. Treat as auth failure.
        raise _auth_failed("Install token missing, expired, or revoked")
    if install.status == ExtensionInstallStatus.BLACKLISTED:
        raise _not_whitelisted()

    # Fire-and-forget last-seen bump. Wrapped in a SAVEPOINT so a failure
    # here can be rolled back without poisoning the surrounding request
    # transaction owned by `get_db`.
    try:
        async with session.begin_nested():
            install_repo = ExtensionInstallRepository(session)
            await install_repo.touch_last_seen(
                install.id, when=datetime.now(timezone.utc)
            )
    except Exception:
        _extension_logger.warning(
            "last_seen_at bump failed for install_id=%s", install.id
        )

    return install


async def require_install(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        extension_bearer_scheme
    ),
    session: AsyncSession = Depends(get_db),
) -> ExtensionInstall:
    """Strict install-token guard for protected extension routes.

    Missing header or non-Bearer scheme → 401 AUTH_FAILED. Valid token
    against a blacklisted install → 403 NOT_WHITELISTED.
    """
    if credentials is None or not credentials.credentials:
        raise _auth_failed("Install token missing")
    if (credentials.scheme or "").lower() != "bearer":
        raise _auth_failed("Install token missing")
    install = await _resolve_install_from_token(
        credentials.credentials, session
    )
    if install is None:
        raise _auth_failed("Install token missing")
    return install


async def optional_install(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        extension_bearer_scheme
    ),
    session: AsyncSession = Depends(get_db),
) -> ExtensionInstall | None:
    """Lenient variant for `GET /api/v1/health` — absent header is fine
    (returns `None`); a present-but-invalid token still 401s per spec."""
    if credentials is None or not credentials.credentials:
        return None
    if (credentials.scheme or "").lower() != "bearer":
        raise _auth_failed("Install token missing")
    return await _resolve_install_from_token(credentials.credentials, session)
