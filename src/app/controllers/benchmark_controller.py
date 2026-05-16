from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from src.core.sse import SSEBroker, SSEResponse
from src.shared.database.pagination import PaginationParams, get_pagination
from src.shared.responses import ApiResponse, PaginatedResponse

from src.app.dependencies import (
    get_benchmark_service,
    get_current_user,
    get_sse_broker,
    require_admin,
    require_authenticated,
)
from src.app.dtos.requests import (
    BenchmarkDatasetCreateRequest,
    BenchmarkRunRequest,
)
from src.app.dtos.responses import (
    BenchmarkDatasetDetail,
    BenchmarkDatasetSummary,
    BenchmarkDetailResponse,
    BenchmarkSummary,
)
from src.app.models.enums import BenchmarkStatus
from src.app.models.user import User
from src.app.services.benchmark_service import BenchmarkService


router = APIRouter(dependencies=[Depends(require_authenticated)])


# ── datasets ──

@router.get("/datasets", dependencies=[Depends(require_admin)])
async def list_datasets(
    pagination: PaginationParams = Depends(get_pagination),
    service: BenchmarkService = Depends(get_benchmark_service),
):
    datasets, total = await service.list_datasets(
        page=pagination.page, page_size=pagination.page_size
    )
    return PaginatedResponse.ok(
        value=[BenchmarkDatasetSummary.from_dataset(d) for d in datasets],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.post("/datasets", dependencies=[Depends(require_admin)])
async def create_dataset(
    name: str = Form(..., min_length=1, max_length=120),
    description: Optional[str] = Form(None, max_length=2000),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    service: BenchmarkService = Depends(get_benchmark_service),
):
    request = BenchmarkDatasetCreateRequest(name=name, description=description)
    payload = await file.read()
    dataset = await service.create_dataset(
        payload=payload,
        filename=file.filename,
        content_type=file.content_type,
        request=request,
        actor=current_user,
    )
    return ApiResponse.ok(
        value=BenchmarkDatasetDetail.from_dataset(dataset),
        message="Benchmark dataset created",
    )


@router.get("/datasets/{dataset_id}", dependencies=[Depends(require_admin)])
async def get_dataset(
    dataset_id: UUID,
    service: BenchmarkService = Depends(get_benchmark_service),
):
    dataset = await service.get_dataset(dataset_id)
    return ApiResponse.ok(value=BenchmarkDatasetDetail.from_dataset(dataset))


@router.delete("/datasets/{dataset_id}", dependencies=[Depends(require_admin)])
async def delete_dataset(
    dataset_id: UUID,
    service: BenchmarkService = Depends(get_benchmark_service),
):
    await service.delete_dataset(dataset_id)
    return ApiResponse.ok(
        value={"id": str(dataset_id)},
        message="Benchmark dataset deleted",
    )


# ── runs ──

@router.post("", dependencies=[Depends(require_admin)])
async def start_run(
    payload: BenchmarkRunRequest,
    current_user: User = Depends(get_current_user),
    service: BenchmarkService = Depends(get_benchmark_service),
):
    benchmark = await service.start_run(payload, actor=current_user)
    results = []  # PENDING row has no per-version results yet
    return ApiResponse.ok(
        value=BenchmarkDetailResponse.from_detail(benchmark, results),
        message="Benchmark run submitted",
    )


@router.get("")
async def list_runs(
    pagination: PaginationParams = Depends(get_pagination),
    status: Optional[BenchmarkStatus] = Query(None),
    dataset_id: Optional[UUID] = Query(None, alias="datasetId"),
    service: BenchmarkService = Depends(get_benchmark_service),
):
    benchmarks, total = await service.list_benchmarks(
        page=pagination.page,
        page_size=pagination.page_size,
        status=status,
        dataset_id=dataset_id,
    )
    return PaginatedResponse.ok(
        value=[BenchmarkSummary.from_benchmark(b) for b in benchmarks],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.get("/{benchmark_id}")
async def get_run(
    benchmark_id: UUID,
    service: BenchmarkService = Depends(get_benchmark_service),
):
    benchmark, results = await service.get_benchmark(benchmark_id)
    return ApiResponse.ok(
        value=BenchmarkDetailResponse.from_detail(benchmark, list(results)),
    )


@router.get("/{benchmark_id}/events", dependencies=[Depends(require_admin)])
async def run_events(
    benchmark_id: UUID,
    broker: SSEBroker = Depends(get_sse_broker),
    service: BenchmarkService = Depends(get_benchmark_service),
):
    # The benchmark must exist; an unknown id never opens a stream that would
    # sit idle forever waiting for an event that can't arrive.
    await service.get_benchmark(benchmark_id)
    return SSEResponse(
        broker=broker,
        topic=service.sse_topic(benchmark_id),
        heartbeat_seconds=service.sse_heartbeat_seconds,
    )
