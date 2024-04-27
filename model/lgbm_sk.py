import typing as t
import gc

import lightgbm as lgb
import pandas as pd
import sklearn.decomposition as decomp
import sklearn.pipeline as pipe
# import sklearn.preprocessing as prep
import sklearn.compose as c

import data
import evaluate as evl
import model.params as params
import settings as s
import utils
from sklearn.model_selection import GridSearchCV

logger = utils.logger
model_tag = utils.ModelTag

SEED = 2024

# decomp.KernelPCA()

param_grid = {
    "boosting_type": ["gbdt"],
    "objective": ["binary"],
    "is_unbalance ": ["true"],
    "max_depth": [20],
    "num_leaves": [100, 300, 600],
    "learning_rate": [0.05],
    "colsample_bytree ": [0.8],
    "subsample ": [0.8],
    "subsample_freq ": [5],
    "n_estimators": [100],

}


def get_pipeline_lgbm_sk() -> pipe.Pipeline:
    model = pipe.Pipeline(
        [
            # ("categories", prep.OneHotEncoder(dtype=pd.CategoricalDtype, max_categories=6)),
            # ("decomposition", decomp.TruncatedSVD(n_components=50)),
            # ("decomposition", decomp.PCA(n_components=0.8)),
            # ("decomposition", decomp.KernelPCA(n_components=50, kernel="rbf")),
            ("clf", lgb.LGBMClassifier(**params.lgb_sk))
        ]
    )

    return model


# def get_pipeline_lgbm_sk() -> pipe.Pipeline:
#     model = lgb.LGBMModel(**params.lgb_sk)
#     return model


def grid_search_cv(model, grid: dict, X: pd.DataFrame, y: pd.DataFrame):
    grid = GridSearchCV(model, param_grid=grid, cv=3, n_jobs=-1, scoring="roc_auc")
    grid.fit(X, y)
    best_params = grid.best_params_
    print("Best params:", best_params)
    print("Best score:", grid.best_score_)
    return grid


def run_gcv(params: dict):
    df = data.load_data()
    # df = df.select_dtypes(exclude=['category', 'datetime64[ns]'])
    x_train, y_train, x_test, y_test = data.split_data(df, 0.7)

    model = lgb.LGBMClassifier()
    grid = grid_search_cv(model, params, x_train, y_train)


def run():
    df = data.load_data()
    # df = df.select_dtypes(exclude=['category', 'datetime64[ns]'])
    x_train, y_train, x_test, y_test = data.split_data(df, 0.7)

    model = lgb.LGBMClassifier(**params.lgb_sk)
    model.fit(x_train, y_train, eval_metric="auc")
    y_pred = model.predict(x_test, num_iteration=model.best_iteration_)
    gc.collect()

    scores = evl.evaluate(y_test, y_pred)
    feature_importance = list(zip(x_train.columns, model.feature_importances_))
    feature_importance = sorted(feature_importance, key=lambda x: x[1])[::-1]

    # save model and results
    model_name = "lgbm-sk"
    save_path = s.PATH.MODEL_OUT.joinpath(f"{model_name}_{s.get_timestamp_str()}")
    save_path.mkdir(exist_ok=True)
    model_results = {
        'model': model_name,
        'path': str(save_path),
        'params': model.get_params(),
        'scores': scores

    }
    tag = model_tag(
        model=model,
        model_name=model_name,
        results=model_results,
        model_path=save_path
    )
    tag.save()


# TODO: explain meaning of low Precision and high Recall - where is the model making mistakes?
# TODO: feature selection
# TODO: StratifiedKFold - "each set contains approximately the same percentage of samples of each target class as the complete set"
# https://scikit-learn.org/stable/modules/cross_validation.html#stratified-shuffle-split

# NOTE: LGBMModel produces better results than LGBMClassifier


if __name__ == "__main__":
    run()
    # run_gcv(param_grid)
