SEED = 2024

# light gradient boosting
lgb = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "is_unbalance": "true",
    "metric": "auc",
    "max_depth": 3,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 1000,
    "verbose": -1,
    "random_state": SEED
}

lgb_sk = {
    "boosting_type": "gbdt",
    "objective": "binary",
    # "class_weight ": "balanced",
    "is_unbalance": "true",
    "max_depth": 20,
    "num_leaves": 500,
    "learning_rate": 0.05,
    "colsample_bytree": 0.7,
    "subsample": 0.7,
    "subsample_freq": 5,
    "n_estimators": 100,
    "n_jobs": -1,
    "random_state": SEED
}

# logistic regression
lg_elnet = {
    "n_jobs": -1,
    "l1_ratio": 0.5,
    "penalty": "elasticnet",
    "solver": "saga",
    "random_state": SEED
}

lg_l2 = {
    "n_jobs": -1,
    "max_iter": 200,
    "random_state": SEED
}

lg_l2_c = {
    "C": 1.3,  # smaller values -> stronger regularization
    "n_jobs": -1,
    "max_iter": 200,
    # "solver": "saga",
    "class_weight": "balanced",
    "random_state": SEED
}

svm_rbf = {
    "kernel": "rbf",
    "gamma": "auto",
    "class_weight": "balanced",
    "random_state": SEED
}

svm_nu = {
    "kernel": "rbf",
    "nu": 0.5,
    "gamma": "auto",
    "class_weight": "balanced",
    "random_state": SEED
}

svm_linear = {
    "kernel": "linear",
    "gamma": "auto",
    "class_weight": "balanced",
    "random_state": SEED
}

random_forest = {
    "n_estimators": 100,
    # "max_depth": 20,
    # "min_samples_leaf": 0.2,
    "max_features": 0.7,
    # "max_leaf_nodes": 100,
    "n_jobs": -1,
    "max_samples": 0.9
}

random_forest_wgt = {
    **random_forest,
    "class_weight": "balanced_subsample"
}
