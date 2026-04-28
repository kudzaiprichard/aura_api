from src.app.controllers.analysis_controller import router as analysis_router
from src.app.controllers.auth_controller import router as auth_router
from src.app.controllers.benchmark_controller import (
    router as benchmark_router,
)
from src.app.controllers.dashboard_controller import (
    router as dashboard_router,
)
from src.app.controllers.drift_controller import router as drift_router
from src.app.controllers.escalation_controller import (
    router as escalation_router,
)
from src.app.controllers.extension_auth_controller import (
    router as extension_auth_router,
)
from src.app.controllers.extension_email_controller import (
    router as extension_email_router,
)
from src.app.controllers.extension_install_controller import (
    router as extension_install_router,
)
from src.app.controllers.model_controller import router as model_router
from src.app.controllers.review_controller import router as review_router
from src.app.controllers.system_controller import router as system_router
from src.app.controllers.training_buffer_controller import (
    router as training_buffer_router,
)
from src.app.controllers.training_run_controller import (
    router as training_run_router,
)
from src.app.controllers.user_controller import router as user_router

__all__ = [
    "analysis_router",
    "auth_router",
    "benchmark_router",
    "dashboard_router",
    "drift_router",
    "escalation_router",
    "extension_auth_router",
    "extension_email_router",
    "extension_install_router",
    "model_router",
    "review_router",
    "system_router",
    "training_buffer_router",
    "training_run_router",
    "user_router",
]
