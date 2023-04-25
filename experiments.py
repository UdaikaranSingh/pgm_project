from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from scipy.stats import entropy
import warnings

from causalml.inference.meta import (
    BaseXRegressor,
    BaseRRegressor,
    BaseSRegressor,
    BaseTRegressor,
)
from causalml.propensity import ElasticNetPropensityModel
from causalml.dataset import simulate_nuisance_and_easy_treatment

from MyRegressor import MyBaggingRegressor

plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")

KEY_GENERATED_DATA = "generated_data"
KEY_ACTUAL = "Actuals"

RANDOM_SEED = 42



def get_synthetic_preds_holdout(
    synthetic_data_func, n=1000, valid_size=0.2, estimators={}
):
    """Generate predictions for synthetic data using specified function (single simulation) for train and holdout
    Args:
        synthetic_data_func (function): synthetic data generation function
        n (int, optional): number of samples
        valid_size(float,optional): validaiton/hold out data size
        estimators (dict of object): dict of names and objects of treatment effect estimators
    Returns:
        (tuple): synthetic training and validation data dictionaries:
          - preds_dict_train (dict): synthetic training data dictionary
          - preds_dict_valid (dict): synthetic validation data dictionary
    """
    y, X, w, tau, b, e = synthetic_data_func(n=n)

    (
        X_train,
        X_val,
        y_train,
        y_val,
        w_train,
        w_val,
        tau_train,
        tau_val,
        b_train,
        b_val,
        e_train,
        e_val,
    ) = train_test_split(
        X, y, w, tau, b, e, test_size=valid_size, random_state=RANDOM_SEED, shuffle=True
    )

    preds_dict_train = {}
    preds_dict_valid = {}

    preds_dict_train[KEY_ACTUAL] = tau_train
    preds_dict_valid[KEY_ACTUAL] = tau_val

    preds_dict_train["generated_data"] = {
        "y": y_train,
        "X": X_train,
        "w": w_train,
        "tau": tau_train,
        "b": b_train,
        "e": e_train,
    }
    preds_dict_valid["generated_data"] = {
        "y": y_val,
        "X": X_val,
        "w": w_val,
        "tau": tau_val,
        "b": b_val,
        "e": e_val,
    }

    # Predict p_hat because e would not be directly observed in real-life
    p_model = ElasticNetPropensityModel()
    p_hat_train = p_model.fit_predict(X_train, w_train)
    p_hat_val = p_model.fit_predict(X_val, w_val)

    for base_learner, label_l in zip(
        [BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor],
        ["S", "T", "X", "R"],
    ):
        for model, label_m in zip([LinearRegression, XGBRegressor, MyBaggingRegressor], ["LR", "XGB", "MBR"]):
            # RLearner will need to fit on the p_hat
            if label_l != "R":
                learner = base_learner(model())
                # fit the model on training data only
                learner.fit(X=X_train, treatment=w_train, y=y_train)
                try:
                    preds_dict_train[
                        "{} Learner ({})".format(label_l, label_m)
                    ] = learner.predict(X=X_train, p=p_hat_train).flatten()
                    preds_dict_valid[
                        "{} Learner ({})".format(label_l, label_m)
                    ] = learner.predict(X=X_val, p=p_hat_val).flatten()
                except TypeError:
                    preds_dict_train[
                        "{} Learner ({})".format(label_l, label_m)
                    ] = learner.predict(
                        X=X_train, treatment=w_train, y=y_train
                    ).flatten()
                    preds_dict_valid[
                        "{} Learner ({})".format(label_l, label_m)
                    ] = learner.predict(X=X_val, treatment=w_val, y=y_val).flatten()
            else:
                learner = base_learner(model())
                learner.fit(X=X_train, p=p_hat_train, treatment=w_train, y=y_train)
                preds_dict_train[
                    "{} Learner ({})".format(label_l, label_m)
                ] = learner.predict(X=X_train).flatten()
                preds_dict_valid[
                    "{} Learner ({})".format(label_l, label_m)
                ] = learner.predict(X=X_val).flatten()

    return preds_dict_train, preds_dict_valid



def get_synthetic_summary_holdout(synthetic_data_func, n=1000, valid_size=0.2, k=1):
    """Generate a summary for predictions on synthetic data for train and holdout using specified function
    Args:
        synthetic_data_func (function): synthetic data generation function
        n (int, optional): number of samples per simulation
        valid_size(float,optional): validation/hold out data size
        k (int, optional): number of simulations
    Returns:
        (tuple): summary evaluation metrics of predictions for train and validation:
          - summary_train (pandas.DataFrame): training data evaluation summary
          - summary_train (pandas.DataFrame): validation data evaluation summary
    """

    summaries_train = []
    summaries_validation = []

    for i in range(k):
        preds_dict_train, preds_dict_valid = get_synthetic_preds_holdout(
            synthetic_data_func, n=n, valid_size=valid_size
        )
        actuals_train = preds_dict_train[KEY_ACTUAL]
        actuals_validation = preds_dict_valid[KEY_ACTUAL]

        synthetic_summary_train = pd.DataFrame(
            {
                label: [preds.mean(), mse(preds, actuals_train)]
                for label, preds in preds_dict_train.items()
                if KEY_GENERATED_DATA not in label.lower()
            },
            index=["ATE", "MSE"],
        ).T
        synthetic_summary_train["Abs % Error of ATE"] = np.abs(
            (
                synthetic_summary_train["ATE"]
                / synthetic_summary_train.loc[KEY_ACTUAL, "ATE"]
            )
            - 1
        )

        synthetic_summary_validation = pd.DataFrame(
            {
                label: [preds.mean(), mse(preds, actuals_validation)]
                for label, preds in preds_dict_valid.items()
                if KEY_GENERATED_DATA not in label.lower()
            },
            index=["ATE", "MSE"],
        ).T
        synthetic_summary_validation["Abs % Error of ATE"] = np.abs(
            (
                synthetic_summary_validation["ATE"]
                / synthetic_summary_validation.loc[KEY_ACTUAL, "ATE"]
            )
            - 1
        )

        # calculate kl divergence for training
        for label in synthetic_summary_train.index:
            stacked_values = np.hstack((preds_dict_train[label], actuals_train))
            stacked_low = np.percentile(stacked_values, 0.1)
            stacked_high = np.percentile(stacked_values, 99.9)
            bins = np.linspace(stacked_low, stacked_high, 100)

            distr = np.histogram(preds_dict_train[label], bins=bins)[0]
            distr = np.clip(distr / distr.sum(), 0.001, 0.999)
            true_distr = np.histogram(actuals_train, bins=bins)[0]
            true_distr = np.clip(true_distr / true_distr.sum(), 0.001, 0.999)

            kl = entropy(distr, true_distr)
            synthetic_summary_train.loc[label, "KL Divergence"] = kl

        # calculate kl divergence for validation
        for label in synthetic_summary_validation.index:
            stacked_values = np.hstack((preds_dict_valid[label], actuals_validation))
            stacked_low = np.percentile(stacked_values, 0.1)
            stacked_high = np.percentile(stacked_values, 99.9)
            bins = np.linspace(stacked_low, stacked_high, 100)

            distr = np.histogram(preds_dict_valid[label], bins=bins)[0]
            distr = np.clip(distr / distr.sum(), 0.001, 0.999)
            true_distr = np.histogram(actuals_validation, bins=bins)[0]
            true_distr = np.clip(true_distr / true_distr.sum(), 0.001, 0.999)

            kl = entropy(distr, true_distr)
            synthetic_summary_validation.loc[label, "KL Divergence"] = kl

        summaries_train.append(synthetic_summary_train)
        summaries_validation.append(synthetic_summary_validation)

    summary_train = sum(summaries_train) / k
    summary_validation = sum(summaries_validation) / k
    return (
        summary_train[["Abs % Error of ATE", "MSE", "KL Divergence"]],
        summary_validation[["Abs % Error of ATE", "MSE", "KL Divergence"]],
    )



if __name__ == '__main__':
    train_summary, validation_summary = get_synthetic_summary_holdout(
        simulate_nuisance_and_easy_treatment,
        n = 10000,
        valid_size = 0.2,
        k = 10
    )

    print(train_summary)
    print(validation_summary)
    train_summary.to_csv('train_summary.csv')
    validation_summary.to_csv('validation_summary.csv')
