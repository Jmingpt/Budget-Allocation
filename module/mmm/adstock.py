import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from scipy.signal import convolve2d
from optuna.integration import OptunaSearchCV
from optuna.distributions import UniformDistribution, IntUniformDistribution
from .mmmTransform import row_to_pivot


class ExponentialSaturation(BaseEstimator, TransformerMixin):
    def __init__(self, a=1.):
        self.a = a

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)  # from BaseEstimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)  # from BaseEstimator
        return 1 - np.exp(-self.a * X)


class ExponentialCarryover(BaseEstimator, TransformerMixin):
    def __init__(self, strength=0.5, length=1):
        self.strength = strength
        self.length = length

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self.sliding_window_ = (
            self.strength ** np.arange(self.length + 1)
        ).reshape(-1, 1)
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        convolution = convolve2d(X, self.sliding_window_)
        if self.length > 0:
            convolution = convolution[: -self.length]
        return convolution


def adstock(df):
    if df is not None:
        _, mmm_df = row_to_pivot(df)
        X = mmm_df.drop('Revenue', axis=1)
        y = mmm_df['Revenue']

        colTranPipeline = []
        for col in X.columns:
            lower_name = col.lower().replace(' ', '_')
            colTranPipeline.append((f'{lower_name}_pipe', Pipeline([('carryover', ExponentialCarryover()), ('saturation', ExponentialSaturation())]), [col]))

        adstock = ColumnTransformer(colTranPipeline, remainder='passthrough')
        model = Pipeline([('adstock', adstock), ('regression', LinearRegression())])
        model.fit(X, y)

        param_dist = {
            'adstock__app_install_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__app_install_pipe__carryover__length': IntUniformDistribution(0, 6),
            'adstock__app_install_pipe__saturation__a': UniformDistribution(0, 0.01),
            'adstock__discovery_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__discovery_pipe__carryover__length': IntUniformDistribution(0, 6),
            'adstock__discovery_pipe__saturation__a': UniformDistribution(0, 0.01),
            'adstock__facebook_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__facebook_pipe__carryover__length': IntUniformDistribution(0, 6),
            'adstock__facebook_pipe__saturation__a': UniformDistribution(0, 0.01),
            'adstock__g_shopping_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__g_shopping_pipe__carryover__length': IntUniformDistribution(0, 6),
            'adstock__g_shopping_pipe__saturation__a': UniformDistribution(0, 0.01),
            'adstock__gdn_non-smart_display_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__gdn_non-smart_display_pipe__carryover__length': IntUniformDistribution(0, 6),
            'adstock__gdn_non-smart_display_pipe__saturation__a': UniformDistribution(0, 0.01),
            'adstock__gdn_smart_display_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__gdn_smart_display_pipe__carryover__length': IntUniformDistribution(0, 6),
            'adstock__gdn_smart_display_pipe__saturation__a': UniformDistribution(0, 0.01),
            'adstock__performance_max_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__performance_max_pipe__carryover__length': IntUniformDistribution(0, 6),
            'adstock__performance_max_pipe__saturation__a': UniformDistribution(0, 0.01),
            'adstock__search_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__search_pipe__carryover__length': IntUniformDistribution(0, 6),
            'adstock__search_pipe__saturation__a': UniformDistribution(0, 0.01),
            'adstock__youtube_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__youtube_pipe__carryover__length': IntUniformDistribution(0, 6),
            'adstock__youtube_pipe__saturation__a': UniformDistribution(0, 0.01),
        }
        tuned_model = OptunaSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_trials=1000,
            cv=TimeSeriesSplit(),
            random_state=0
        )
        tuned_model.fit(X, y)
        y_pred = tuned_model.predict(X)
