"""
Global error handlers for FastAPI application.
Catches all exceptions and returns consistent API responses.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from src.shared.responses.api_response import ApiResponse, ErrorDetail
from src.shared.exceptions.exceptions import AppException

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:

    @app.exception_handler(AppException)
    async def handle_app_exception(_req: Request, exc: AppException):
        response = ApiResponse.failure(error=exc.error_detail, message=exc.message)
        return JSONResponse(
            status_code=exc.error_detail.status,
            content=response.model_dump(exclude_none=True, by_alias=True),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(_req: Request, exc: RequestValidationError):
        builder = ErrorDetail.builder("Validation Failed", "VALIDATION_ERROR", 400)
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            builder.add_field_error(field, error["msg"])
        response = ApiResponse.failure(
            error=builder.build(),
            message="Please check your input and try again",
        )
        return JSONResponse(status_code=400, content=response.model_dump(exclude_none=True, by_alias=True))

    @app.exception_handler(ValidationError)
    async def handle_pydantic_validation_error(_req: Request, exc: ValidationError):
        builder = ErrorDetail.builder("Validation Failed", "VALIDATION_ERROR", 400)
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            builder.add_field_error(field, error["msg"])
        response = ApiResponse.failure(
            error=builder.build(),
            message="Please check your input and try again",
        )
        return JSONResponse(status_code=400, content=response.model_dump(exclude_none=True, by_alias=True))

    @app.exception_handler(IntegrityError)
    async def handle_integrity_error(_req: Request, exc: IntegrityError):
        logger.warning("IntegrityError: %s", exc)
        error = ErrorDetail(
            title="Conflict",
            code="CONFLICT",
            status=409,
            details=["A record with the supplied unique values already exists"],
        )
        response = ApiResponse.failure(
            error=error,
            message="This record already exists",
        )
        return JSONResponse(
            status_code=409,
            content=response.model_dump(exclude_none=True, by_alias=True),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(req: Request, exc: StarletteHTTPException):
        user_messages = {
            400: "Please check your request and try again",
            401: "Please log in to continue",
            403: "You don't have permission to perform this action",
            404: "The page you're looking for doesn't exist",
            405: "This action is not allowed",
            429: "Too many requests. Please try again later",
            500: "Something went wrong. Please try again later",
            503: "The service is temporarily unavailable",
        }
        code_by_status = {
            400: "BAD_REQUEST",
            401: "UNAUTHENTICATED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            409: "CONFLICT",
            422: "VALIDATION_ERROR",
            429: "RATE_LIMITED",
            500: "INTERNAL_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }
        title_by_status = {
            400: "Bad Request",
            401: "Unauthenticated",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            409: "Conflict",
            422: "Validation Failed",
            429: "Too Many Requests",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }
        detail_messages = {
            404: f"{req.method} {req.url.path} was not found",
            405: f"{req.method} is not allowed for {req.url.path}",
        }
        error = ErrorDetail(
            title=title_by_status.get(exc.status_code, "Error"),
            code=code_by_status.get(exc.status_code, "HTTP_ERROR"),
            status=exc.status_code,
            details=[detail_messages.get(exc.status_code, str(exc.detail))],
        )
        response = ApiResponse.failure(
            error=error,
            message=user_messages.get(exc.status_code, "An error occurred. Please try again"),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=response.model_dump(exclude_none=True, by_alias=True),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_req: Request, exc: Exception):
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        error = ErrorDetail(
            title="Internal Server Error",
            code="INTERNAL_ERROR",
            status=500,
            details=["An unexpected error occurred. Please try again later."],
        )
        response = ApiResponse.failure(
            error=error,
            message="Something went wrong. Please try again later",
        )
        return JSONResponse(status_code=500, content=response.model_dump(exclude_none=True, by_alias=True))
