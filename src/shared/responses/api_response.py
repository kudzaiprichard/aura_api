from __future__ import annotations

from typing import Generic, TypeVar, Optional, List, Dict
from pydantic import BaseModel, model_validator, Field
from math import ceil

T = TypeVar("T")


# ──────────────────────────────────────────────
# ErrorDetail
# ──────────────────────────────────────────────

class ErrorDetail(BaseModel):
    title: str
    code: str
    status: int
    details: Optional[List[str]] = Field(default=None)
    field_errors: Optional[Dict[str, List[str]]] = Field(
        default=None, alias="fieldErrors"
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {},
    }

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("by_alias", True)
        return super().model_dump_json(**kwargs)

    class Builder:
        def __init__(self, title: str, code: str, status: int):
            self._title = title
            self._code = code
            self._status = status
            self._details: List[str] = []
            self._field_errors: Dict[str, List[str]] = {}

        def add_detail(self, detail: str) -> ErrorDetail.Builder:
            self._details.append(detail)
            return self

        def add_field_error(self, field: str, error: str) -> ErrorDetail.Builder:
            self._field_errors.setdefault(field, []).append(error)
            return self

        def add_field_errors(self, field: str, errors: List[str]) -> ErrorDetail.Builder:
            self._field_errors.setdefault(field, []).extend(errors)
            return self

        def build(self) -> ErrorDetail:
            return ErrorDetail(
                title=self._title,
                code=self._code,
                status=self._status,
                details=self._details if self._details else None,
                field_errors=self._field_errors if self._field_errors else None,
            )

    @staticmethod
    def builder(title: str, code: str, status: int) -> ErrorDetail.Builder:
        return ErrorDetail.Builder(title, code, status)

    def has_details(self) -> bool:
        return bool(self.details)

    def has_field_errors(self) -> bool:
        return bool(self.field_errors)


# ──────────────────────────────────────────────
# ApiResponse
# ──────────────────────────────────────────────

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    message: Optional[str] = None
    value: Optional[T] = None
    error: Optional[ErrorDetail] = None

    model_config = {"populate_by_name": True}

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("by_alias", True)
        return super().model_dump_json(**kwargs)

    @model_validator(mode="after")
    def validate_exclusive(self):
        if self.error is not None and self.value is not None:
            raise ValueError("ApiResponse cannot have both error and value")
        return self

    @staticmethod
    def ok(value: T, message: Optional[str] = None) -> ApiResponse[T]:
        return ApiResponse(success=True, message=message, value=value)

    @staticmethod
    def failure(error: ErrorDetail, message: Optional[str] = None) -> ApiResponse[None]:
        return ApiResponse(success=False, message=message, error=error)


# ──────────────────────────────────────────────
# PaginatedResponse
# ──────────────────────────────────────────────

class PaginationInfo(BaseModel):
    page: int
    total: int
    page_size: int = Field(alias="pageSize")
    total_pages: int = Field(default=0, alias="totalPages")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def compute_total_pages(self):
        if self.page_size <= 0:
            raise ValueError("Page size must be greater than 0")
        if self.total_pages == 0:
            self.total_pages = ceil(self.total / self.page_size)
        return self


class PaginatedResponse(ApiResponse[List[T]], Generic[T]):
    pagination: Optional[PaginationInfo] = None

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("by_alias", True)
        return super().model_dump_json(**kwargs)

    @staticmethod
    def ok(
        value: List[T],
        page: int,
        total: int,
        page_size: int,
        message: Optional[str] = None,
    ) -> PaginatedResponse[T]:
        pagination = PaginationInfo(page=page, total=total, pageSize=page_size)
        return PaginatedResponse(
            success=True,
            message=message,
            value=value,
            pagination=pagination,
        )