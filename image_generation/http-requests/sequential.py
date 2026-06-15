import math
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Sequential(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        self.total_features_ = X.shape[1]
        self.image_side_ = math.ceil(math.sqrt(self.total_features_))   
        image_len = self.image_side_ * self.image_side_
        self.padding_needed_ = image_len - self.total_features_  

        return self

    def transform(self, X):
        padded_data = np.pad(X, ((0, 0), (0, self.padding_needed_)), 'constant')
        image_data = padded_data.reshape(X.shape[0], self.image_side_, self.image_side_)
        return image_data

