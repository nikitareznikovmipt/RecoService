import pandas as pd
import numpy as np
import scipy as sp
from implicit.nearest_neighbours import CosineRecommender
from collections import Counter
import os
from models.top_popular import TopPopular

root_dit = os.path.dirname(os.getcwd())


class UserKnnWithColdStart:
    def __init__(self, N_users: int = 50, N_recomends: int = 10) -> None:
        self.model = CosineRecommender(N_users)
        self.N_users = N_users
        self.N_recomends = N_recomends
        self.is_fitted = False

    def fit(
            self, 
            interactions: pd.DataFrame, 
            user_col: str = 'user_id',
            item_col: str = 'item_id',
            weight_col: str = 'weight'
    ) -> None:
        # Initialize mapping and sparce matrix
        self._create_mapping(interactions, user_col, item_col)
        sparce_matrix = self._create_sparce_matrix(interactions, user_col, item_col, weight_col)
        # Fit user knn if need
        self.model.fit(sparce_matrix)
        # Calculate cold start
        popular_model = TopPopular()
        popular_model.fit(interactions, item_col)
        self.top_popular = popular_model.recommend()
        # Calculate user_to_item interactions
        self.user_to_item = self._get_items_by_user(interactions, user_col, item_col)
        # Calculate idf per item
        self.item_idf = self._calculate_item_idf(interactions, item_col)
        self.is_fitted = True


    def _calculate_item_idf(self, dataset: pd.DataFrame, item_col: str):
        item_cnt = Counter(dataset[item_col].values)
        gen_cnt = sum(item_cnt.values())
        item_idf = {item: np.log(gen_cnt / (cnt + 1)) for item, cnt in item_cnt.items()}
        return item_idf


    def _create_mapping(self, dataset: pd.DataFrame, user_col: str, item_col: str) -> None:
        self.user_mapping = {user: n for n, user in enumerate(dataset[user_col].unique())}
        self.user_inv_mapping = {n: user for user, n in self.user_mapping.items()}

        self.item_mapping = {item: n for n, item in enumerate(dataset[item_col].unique())}
        self.item_inv_mapping = {n: item for item, n in self.item_mapping.items()}
        return self
    

    def _create_sparce_matrix(
            self,
            dataset: pd.DataFrame,
            user_col: str,
            item_col: str,
            weight_col: str
    ) -> None:
        if weight_col is None:
            weights = np.ones(dataset.shape[0], dtype=np.float32)
        else:
            weights = dataset[weight_col].astype(np.float32)
        cols = dataset[user_col].map(self.user_mapping)
        rows = dataset[item_col].map(self.item_mapping)
        sparce_matrix = sp.sparse.csr_matrix(
            (weights, (rows, cols)),
            shape=(dataset[item_col].nunique(), dataset[user_col].nunique())
        )
        return sparce_matrix


    def _get_items_by_user(self, dataset: pd.DataFrame, user_col: str, item_col: str) -> None:
        history_interactions = dataset.groupby(user_col, as_index=False).agg({item_col: list}).to_dict('records')
        user_to_item = {self.user_mapping[elem[user_col]]: elem[item_col] for elem in history_interactions}
        return user_to_item
    

    def recommend(self, user_id: int) -> None:
        if not self.is_fitted:
            raise ValueError('call fit method')
        
        # Check is user cold
        if user_id not in self.user_mapping:
            return self.top_popular[:self.N_recomends]
        
        recs_items = []
        item_scores = []
        map_user_id = self.user_mapping[user_id]
        # Get simularity users
        sim_users, user_scores = self.model.similar_items(map_user_id, N=self.N_users)
        # Remove self user id
        sim_users = sim_users[1:]
        user_scores = user_scores[1:]
        
        for sim_user, score in zip(sim_users, user_scores):
            for item in self.user_to_item[sim_user]:
                recs_items.append(item)
                item_scores.append(self.item_idf[item] * score)
        
        # Sort items by relevance
        sorted_indexes = np.argsort(item_scores)[::-1][:self.N_recomends]
        recs_items = list(np.array(recs_items)[sorted_indexes])
        if len(recs_items) < self.N_recomends:
            need_objects_cnt = self.N_recomends - len(recs_items)
            added_cnt = 0
            item_index = 0
            while added_cnt != need_objects_cnt:
                item = self.top_popular[item_index]
                item_index += 1
                if item not in recs_items:
                    recs_items.append(item)
                    added_cnt += 1
        return recs_items

