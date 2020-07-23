import numpy as np
from sklearn.decomposition import PCA

class PrincipalComponentAnalysis:
    def __init__(self):
        pass
    def shape_features(self, path):
        with open(path, "rb") as f:
            self.features = np.load(f)
        return self.features.shape

    def shape_decomposed_features(self, dim):
        self.pca = PCA(n_components=dim)
        self.pca.fit(self.features)
        self.decomposed_features = self.pca.transform(self.features)
        return self.decomposed_features.shape

    def explain_variance_ratio(self):
        return self.pca.explained_variance_ratio_
