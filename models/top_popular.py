import pandas as pd

class TopPopular():
    def __init__(self) -> None:
        self.is_fitted = False
    
    def fit(self, dataset: pd.DataFrame, item_col: str):
        self.top_popular = dataset.groupby(item_col).size().sort_values(ascending=False).reset_index()[item_col].values
        self.is_fitted = True
        return self

    def recommend(self):
        if self.is_fitted:
            return self.top_popular
        else:
            raise ValueError('call fit method')

