from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile

from src.shared.database.pagination import PaginationParams, get_pagination
from src.shared.responses import ApiResponse, PaginatedResponse

from src.app.dependencies import (
    get_current_user,
    get_training_buffer_service,
    require_admin,
    require_authenticated,
)
from src.app.dtos.responses import (
    BufferImportSummaryResponse,
    BufferItemResponse,
    BufferStatusResponse,
)
from src.app.models.enums import TrainingBufferSource
from src.app.models.user import User
from src.app.services.training_buffer_service import TrainingBufferService


router = APIRouter(dependencies=[Depends(require_authenticated)])


@router.get("/status")
async def get_status(
    service: TrainingBufferService = Depends(get_training_buffer_service),
):
    status = await service.status()
    return ApiResponse.ok(value=BufferStatusResponse.from_status(status))


@router.get("", dependencies=[Depends(require_admin)])
async def list_items(
    pagination: PaginationParams = Depends(get_pagination),
    label: Optional[int] = Query(None, ge=0, le=1),
    source: Optional[TrainingBufferSource] = Query(None),
    category: Optional[str] = Query(None, max_length=64),
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    service: TrainingBufferService = Depends(get_training_buffer_service),
):
    items, total = await service.list(
        page=pagination.page,
        page_size=pagination.page_size,
        label=label,
        source=source,
        category=category,
        date_from=date_from,
        date_to=date_to,
    )
    return PaginatedResponse.ok(
        value=[BufferItemResponse.from_item(i) for i in items],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.get("/{item_id}", dependencies=[Depends(require_admin)])
async def get_item(
    item_id: UUID,
    service: TrainingBufferService = Depends(get_training_buffer_service),
):
    item = await service.get(item_id)
    return ApiResponse.ok(value=BufferItemResponse.from_item(item))


@router.post("/import", dependencies=[Depends(require_admin)])
async def import_csv(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    service: TrainingBufferService = Depends(get_training_buffer_service),
):
    payload = await file.read()
    summary = await service.import_csv(
        payload=payload,
        filename=file.filename,
        content_type=file.content_type,
        actor=current_user,
    )
    return ApiResponse.ok(
        value=BufferImportSummaryResponse.from_summary(summary),
        message="CSV import processed",
    )


@router.delete("/{item_id}", dependencies=[Depends(require_admin)])
async def delete_item(
    item_id: UUID,
    service: TrainingBufferService = Depends(get_training_buffer_service),
):
    await service.delete(item_id)
    return ApiResponse.ok(value={"id": str(item_id)}, message="Buffer item deleted")
