from fastapi import APIRouter, Depends, Query, Response

from src.shared.responses import ApiResponse
from src.app.dependencies import (
    get_current_user,
    get_dashboard_service,
    require_admin,
    require_authenticated,
)
from src.app.models.user import User
from src.app.services.dashboard_service import (
    AdminDashboardQuery,
    AnalystDashboardQuery,
    DashboardService,
)


router = APIRouter(dependencies=[Depends(require_authenticated)])


# §9.4: responses are cacheable per caller for a short TTL. `private` keeps
# them out of shared caches (the payload is scoped to the caller's role or
# identity); `max-age=10` absorbs the tight poll cadence the admin UI uses.
_CACHE_CONTROL = "private, max-age=10"


@router.get("/admin", dependencies=[Depends(require_admin)])
async def admin_dashboard(
    response: Response,
    recent_limit: int = Query(20, alias="recentLimit", ge=0, le=200),
    activity_limit: int = Query(20, alias="activityLimit", ge=0, le=200),
    breakdown_days: int = Query(14, alias="breakdownDays", ge=1, le=90),
    volume_bucket: str = Query("day", alias="volumeBucket"),
    volume_range_hours: int = Query(
        168, alias="volumeRangeHours", ge=1, le=24 * 90
    ),
    sla_seconds: int | None = Query(None, alias="slaSeconds", ge=1),
    timezone_name: str = Query("UTC", alias="timezone"),
    service: DashboardService = Depends(get_dashboard_service),
):
    response.headers["Cache-Control"] = _CACHE_CONTROL
    payload = await service.admin_snapshot(
        AdminDashboardQuery(
            recent_limit=recent_limit,
            activity_limit=activity_limit,
            breakdown_days=breakdown_days,
            volume_bucket=volume_bucket,
            volume_range_hours=volume_range_hours,
            sla_seconds=sla_seconds,
            timezone=timezone_name,
        )
    )
    return ApiResponse.ok(
        value=payload.model_dump(exclude_none=False, by_alias=True),
    )


@router.get("/me")
async def analyst_dashboard(
    response: Response,
    recent_limit: int = Query(20, alias="recentLimit", ge=0, le=200),
    invocation_limit: int = Query(
        10, alias="invocationLimit", ge=0, le=200
    ),
    range_days: int = Query(30, alias="rangeDays", ge=1, le=365),
    sla_seconds: int | None = Query(None, alias="slaSeconds", ge=1),
    timezone_name: str = Query("UTC", alias="timezone"),
    current_user: User = Depends(get_current_user),
    service: DashboardService = Depends(get_dashboard_service),
):
    response.headers["Cache-Control"] = _CACHE_CONTROL
    payload = await service.analyst_snapshot(
        caller=current_user,
        query=AnalystDashboardQuery(
            recent_limit=recent_limit,
            invocation_limit=invocation_limit,
            range_days=range_days,
            sla_seconds=sla_seconds,
            timezone=timezone_name,
        ),
    )
    return ApiResponse.ok(
        value=payload.model_dump(exclude_none=False, by_alias=True),
    )
