import pandas as pd
import numpy as np
import imblearn
from imblearn.over_sampling import ADASYN, SMOTENC

def resample(X, y, SEED=386, ratio=None):
    '''
    X: dataframe
    y: dataframe
    ratio: float = N_minority / N_marjority. If not specified, auto
    
    return: X_resampled, y_resampled
    '''
    
    cat_cols = X.select_dtypes(include="category").columns.tolist()
    
    if cat_cols:
        if ratio:
            X_resampled, y_resampled = SMOTENC(random_state=SEED, ratio=ratio, categorical_features=cat_cols).fit_sample(X, y)
        else:
            X_resampled, y_resampled = SMOTENC(random_state=SEED, categorical_features=cat_cols).fit_sample(X, y)
    else:
        if ratio:
            X_resampled, y_resampled = SMOTENC(random_state=SEED, ratio=ratio).fit_sample(X, y)
        else:
            X_resampled, y_resampled = SMOTENC(random_state=SEED).fit_sample(X, y)
    
    return X_resampled, y_resampled