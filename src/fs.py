from sklearn.feature_selection import mutual_info_regression
from feature_engine.selection import DropHighPSIFeatures, RecursiveFeatureAddition
import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def mi_features(X, y):
    result = pd.Series(mutual_info_regression(X = X, y = y), index = X.columns)
    result.name = 'Mutual_Info'
    return result


def psi_features(X,y,split_frac):
    transformer = DropHighPSIFeatures(split_frac = split_frac)
    transformer.fit(X)
    result = pd.Series(transformer.psi_values_)
    result.name = 'PSI'
    return result

def rfa_features(X,y, scoring, cv, model):
    model_name = type(model).__name__   
    tr = RecursiveFeatureAddition(estimator = model, scoring = scoring, cv = cv)
    tr.fit(X,y)
    result = pd.Series(tr.performance_drifts_)
    result.name = f'Feature Addition {model_name}'
    return result
    