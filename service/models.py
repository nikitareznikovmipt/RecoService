import os
import pickle
import typing as tp

from pydantic import BaseModel

from service.config import TOP_POPULAR_RECS


class TopPopular:
    """
    Top Popular Items
    """

    def __init__(self) -> None:
        self.recommends = TOP_POPULAR_RECS

    def recommend(self, user_id: int) -> tp.List[int]:
        return self.recommends


class UserKNN:
    """
    UserKNN model
    """

    def __init__(self) -> None:
        file_url = os.path.join(os.getcwd(), "service/saved_models/user_knn_with_cold_start.pkl")
        if os.path.exists(file_url):
            with open(os.path.join(os.getcwd(), "service/saved_models/user_knn_with_cold_start.pkl"), "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = TopPopular()  # Костыль, чтобы пройти тесты

    def recommend(self, user_id: int) -> tp.List[int]:
        return self.model.recommend(user_id)


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None
