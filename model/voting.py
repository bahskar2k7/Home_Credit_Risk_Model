import sklearn.ensemble as ens

import data
# from sklearn.model_selection import cross_val_score
import evaluate as evl
import model.lgbm_sk as lgbm
import model.logistic_regression as log_reg
import model.random_forest as rf
import settings as s
import utils

logger = utils.logger
model_tag = utils.ModelTag
timeit = utils.time_this


# https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting

@timeit
def run():
    clf1 = log_reg.get_pipeline_logistic_regression()
    clf2 = log_reg.get_pipeline_logistic_regression_l2()
    clf3 = lgbm.get_pipeline_lgbm_sk()
    clf4 = rf.get_pipeline_random_forest_wgt()

    model = ens.VotingClassifier(
        estimators=[
            ("logistic-regression-1", clf1),
            ("logistic-regression-2", clf2),
            ("lgbm", clf3),
            ("rf-2", clf4),
        ],
        voting="soft",
        weights=[1, 1, 2, 2],
        # n_jobs=-1
    )

    df = data.load_data()
    df = df.select_dtypes(exclude=['category', 'datetime64[ns]'])

    x_train, y_train, x_test, y_test = data.split_data(df, 0.7)
    # scores = cross_val_score(model, x_train, y_train, cv=3, scoring="roc_auc")
    # mean_scores = scores.mean()
    # print("Scores:\n", mean_scores)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores = evl.evaluate(y_test, y_pred)

    print(scores)

    # save model and results
    model_name = "voting-clf"
    model_params = model.get_params()

    save_path = s.PATH.MODEL_OUT.joinpath(f"{model_name}_{s.get_timestamp_str()}")
    save_path.mkdir(exist_ok=True)
    model_results = {
        'model': model_name,
        'path': str(save_path),
        'params': model_params,
        'scores': scores

    }
    tag = model_tag(
        model=model,
        model_name=model_name,
        results=model_results,
        model_path=save_path
    )
    tag.save()


if __name__ == "__main__":
    run()
