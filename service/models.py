import typing as tp

from pydantic import BaseModel

from .config import TOP_POPULAR_RECS


class TopKPopular:
    def __init__(self) -> None:
        self.name = "top_popular"
        self.recomendations: tp.List[int] = []

    def recomend(self) -> tp.List[int]:
        """
        Fit model and recomend top popular recomendations
        """
        recs = TOP_POPULAR_RECS
        return recs


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None
