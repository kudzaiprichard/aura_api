from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.review_disagreement import ReviewDisagreement
from src.shared.database import BaseRepository


class ReviewDisagreementRepository(BaseRepository[ReviewDisagreement]):
    def __init__(self, session: AsyncSession):
        super().__init__(ReviewDisagreement, session)
