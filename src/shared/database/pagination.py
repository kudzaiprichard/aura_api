from dataclasses import dataclass
from fastapi import Query


@dataclass
class PaginationParams:
    page: int
    page_size: int

    @property
    def skip(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


def get_pagination(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize", description="Items per page"),
) -> PaginationParams:
    return PaginationParams(page=page, page_size=page_size)
