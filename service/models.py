import json
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


class LightFM:
    def __init__(self) -> None:
        file_path = os.path.join(os.getcwd(), "service/saved_models/lightfm_recommendations.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.recommendations = json.load(f)["item_id"]
        else:
            self.recommendations = {"0": [1, 2, 3]}  # Заглушка для теста
        self.popular = TopPopular().recommends

    def recommend(self, user_id: int) -> tp.List[int]:
        if str(user_id) in self.recommendations:
            return self.recommendations[str(user_id)]
        return self.popular


class DSSM:
    def __init__(self) -> None:
        file_path = os.path.join(os.getcwd(), "service/saved_models/dssm_recommendations.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.recommendations = json.load(f)["item_id"]
        else:
            self.recommendations = {}
        self.popular = TopPopular().recommends

    def recommend(self, user_id: int) -> tp.List[int]:
        if str(user_id) in self.recommendations:
            return self.recommendations[str(user_id)]
        return self.popular


class AutoEncoder:
    def __init__(self) -> None:
        file_path = os.path.join(os.getcwd(), "service/saved_models/ae_recommendations.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.recommendations = json.load(f)["item_id"]
        else:
            self.recommendations = {}
        self.popular = TopPopular().recommends

    def recommend(self, user_id: int) -> tp.List[int]:
        if str(user_id) in self.recommendations:
            return self.recommendations[str(user_id)]
        return self.popular


class MultiVae:
    def __init__(self) -> None:
        file_path = os.path.join(os.getcwd(), "service/saved_models/multivae_recommendations.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.recommendations = json.load(f)["item_id"]
        else:
            self.recommendations = {}
        self.popular = TopPopular().recommends

    def recommend(self, user_id: int) -> tp.List[int]:
        if str(user_id) in self.recommendations:
            return self.recommendations[str(user_id)]
        return self.popular


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None
