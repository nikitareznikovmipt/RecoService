import typing as tp

import pandas as pd
from pydantic import BaseModel
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import PopularModel


class TopKPopular:
    def __init__(self) -> None:
        self.name = "top_popular"
        self.recomendations: tp.List[int] = []

    def recomend(self) -> tp.List[int]:
        """
        Fit model and recomend top popular recomendations
        """
        if len(self.recomendations) == 0:
            interactions = pd.read_csv("./data/interactions.csv")[["user_id", "item_id", "last_watch_dt", "total_dur"]]
            interactions.columns = [Columns.User, Columns.Item, Columns.Datetime, Columns.Weight]
            dataset = Dataset.construct(interactions)
            model = PopularModel()
            model.fit(dataset)
            recs = model.recommend(users=[602509], dataset=dataset, k=10, filter_viewed=False)
            self.recomendations = list(recs["item_id"].values.astype(int))
        recs = self.recomendations
        return recs


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None
