import typing as t

import sklearn.pipeline as pipe

import data
import evaluate as evl
import model.logistic_regression as log_reg
import model.random_forest as rf
# import model.svm as svm
import model.lgbm_sk as lgbm
import settings as s
import utils

logger = utils.logger
model_tag = utils.ModelTag
timeit = utils.time_this


@timeit
def run(model_name: str, pipeline_callable: t.Callable):
    # https://towardsdatascience.com/a-brief-introduction-to-scikit-pipelines-888edc86da9b
    df = data.load_data()

    # need numeric variables
    df = df.select_dtypes(exclude=['category', 'datetime64[ns]'])

    x_train, y_train, x_test, y_test = data.split_data(df, 0.7)

    model: pipe.Pipeline = pipeline_callable()

    logger.info("Start training model - %s", model_name)
    model.fit(x_train, y_train)

    logger.info("Making predictions")
    y_pred = model.predict(x_test)

    scores = evl.evaluate(y_test, y_pred)
    model_params = model.get_params()

    # save model and results
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


if __name__ == '__main__':
    # run("logistic-regression-wgt", log_reg.get_pipeline_logistic_regression)
    # run("logistic-regression-l2", log_reg.get_pipeline_logistic_regression_l2)
    # run("lgbm-sk", lgbm.get_pipeline_lgbm_sk)
    # run("random-forest", rf.get_pipeline_random_forest)
    run("random-forest-2", rf.get_pipeline_random_forest_wgt)
    # run("svm-rbf", svm.get_pipeline_svm_rbf)
