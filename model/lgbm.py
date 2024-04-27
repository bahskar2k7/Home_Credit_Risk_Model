import gc

import lightgbm as lgb

import data
import evaluate as evl
import model.params as params
import settings as s
import utils

logger = utils.logger
model_tag = utils.ModelTag

SEED = 2024


# TODO: use sklearn style LGBM to acoomodate Voting Classifier ensemble

# lgb.LGBMClassifier()

def run():
    df = data.load_data()
    x_train, y_train, x_test, y_test = data.split_data(df, 0.7)

    # train LGBM
    lgb_train = lgb.Dataset(x_train, label=y_train)
    lgb_test = lgb.Dataset(x_test, label=y_test, reference=lgb_train)

    model = lgb.train(
        params.lgb,
        lgb_train,
        valid_sets=lgb_test,
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(8)]
    )
    y_pred = model.predict(x_test, num_iteration=model.best_iteration)
    gc.collect()

    scores = evl.evaluate(y_test, y_pred)

    # save model and results
    model_name = "lgbm"
    save_path = s.PATH.MODEL_OUT.joinpath(f"{model_name}_{s.get_timestamp_str()}")
    save_path.mkdir(exist_ok=True)
    model_results = {
        'model': model_name,
        'path': str(save_path),
        'params': model.params,
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


if __name__ == "__main__":
    run()
