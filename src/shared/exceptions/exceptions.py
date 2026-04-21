"""
Exception hierarchy for API responses.
Each exception wraps an ErrorDetail object for consistent error handling.
"""

from typing import Optional
from src.shared.responses.api_response import ErrorDetail


class AppException(Exception):
    def __init__(self, message: str = "An error occurred", error_detail: Optional[ErrorDetail] = None):
        self.message = message
        self.error_detail = error_detail or ErrorDetail(
            title="Application Error",
            code="APP_ERROR",
            status=500,
            details=[message] if message else [],
        )
        super().__init__(self.message)


class NotFoundException(AppException):
    def __init__(self, message: str = "The requested resource was not found", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(title="Not Found", code="NOT_FOUND", status=404, details=[message])
        super().__init__(message, error_detail)


class ValidationException(AppException):
    def __init__(self, message: str = "Please check your input and try again", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(title="Validation Failed", code="VALIDATION_ERROR", status=400, details=[message])
        super().__init__(message, error_detail)


class AuthenticationException(AppException):
    def __init__(self, message: str = "Please log in to continue", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(title="Authentication Failed", code="AUTH_FAILED", status=401, details=[message])
        super().__init__(message, error_detail)


class AuthorizationException(AppException):
    def __init__(self, message: str = "You don't have permission to perform this action", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(title="Access Denied", code="FORBIDDEN", status=403, details=[message])
        super().__init__(message, error_detail)


class ConflictException(AppException):
    def __init__(self, message: str = "This resource already exists", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(title="Conflict", code="CONFLICT", status=409, details=[message])
        super().__init__(message, error_detail)


class BadRequestException(AppException):
    def __init__(self, message: str = "Your request could not be processed", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(title="Bad Request", code="BAD_REQUEST", status=400, details=[message])
        super().__init__(message, error_detail)


class InternalServerException(AppException):
    def __init__(self, message: str = "Something went wrong. Please try again later", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(title="Internal Server Error", code="INTERNAL_ERROR", status=500, details=[message])
        super().__init__(message, error_detail)


class ServiceUnavailableException(AppException):
    def __init__(self, message: str = "The service is temporarily unavailable. Please try again later", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(title="Service Unavailable", code="SERVICE_UNAVAILABLE", status=503, details=[message])
        super().__init__(message, error_detail)
