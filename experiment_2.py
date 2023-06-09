from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from scipy.stats import entropy
import warnings
import pickle

from causalml.inference.meta import (
    BaseXRegressor,
    BaseRRegressor,
    BaseSRegressor,
    BaseTRegressor,
)
from causalml.propensity import ElasticNetPropensityModel
from causalml.dataset import simulate_nuisance_and_easy_treatment

from MyRegressor import *
import os

plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")

KEY_GENERATED_DATA = "generated_data"
KEY_ACTUAL = "Actuals"

RANDOM_SEED = 42
LOAD_DATA = False

if not os.path.exists('./results/experiment_2/'):
    os.mkdir('./results/experiment_2/')


def distr_plot_single_sim(
    synthetic_preds,
    savepath,
    kind="kde",
    drop_learners=[],
    bins=50,
    histtype="step",
    alpha=1,
    linewidth=1,
    bw_method=1,
):
    """Plots the distribution of each learner's predictions (for a single simulation).
    Kernel Density Estimation (kde) and actual histogram plots supported.
    Args:
        synthetic_preds (dict): dictionary of predictions generated by get_synthetic_preds()
        kind (str, optional): 'kde' or 'hist'
        drop_learners (list, optional): list of learners (str) to omit when plotting
        bins (int, optional): number of bins to plot if kind set to 'hist'
        histtype (str, optional): histogram type if kind set to 'hist'
        alpha (float, optional): alpha (transparency) for plotting
        linewidth (int, optional): line width for plotting
        bw_method (float, optional): parameter for kde
    """
    preds_for_plot = synthetic_preds.copy()

    # deleted generated data and assign actual value
    del preds_for_plot[KEY_GENERATED_DATA]
    global_lower = np.percentile(np.hstack(preds_for_plot.values()), 1)
    global_upper = np.percentile(np.hstack(preds_for_plot.values()), 99)
    learners = list(preds_for_plot.keys())
    learners = [learner for learner in learners if learner not in drop_learners]

    # Plotting
    plt.figure(figsize=(12, 8))
    colors = [
        "black",
        "red",
        "blue",
        "green",
        "cyan",
        "brown",
        "grey",
        "pink",
        "orange",
        "yellow",
    ]
    for i, (k, v) in enumerate(preds_for_plot.items()):
        if k in learners:
            if kind == "kde":
                v = pd.Series(v.flatten())
                v = v[v.between(global_lower, global_upper)]
                v.plot(
                    kind="kde",
                    bw_method=bw_method,
                    label=k,
                    linewidth=linewidth,
                    color=colors[i],
                )
            elif kind == "hist":
                plt.hist(
                    v,
                    bins=np.linspace(global_lower, global_upper, bins),
                    label=k,
                    histtype=histtype,
                    alpha=alpha,
                    linewidth=linewidth,
                    color=colors[i],
                )
            else:
                pass

    plt.xlim(global_lower, global_upper)
    #plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.title("Distribution from a Single Simulation")
    plt.savefig(savepath)

def plot_dist(dictionary, linear_models, save_path, alpha = 0.2, bins = 30):
    plt.figure(figsize=(12,8))
    keys = dictionary.keys()
    lin_models = linear_models
    not_lin_models = [key for key in keys if key not in linear_models]
    for k in not_lin_models:
        values = dictionary[k]
        plt.hist(values, alpha = alpha, bins = bins, label = k, range=[-0.5, 2])
    for k in lin_models:
        values = dictionary[k]
        plt.axvline(values[0], label=k,
           linestyle='dotted', color=np.random.rand(3,), linewidth=2)
    plt.title('Distribution of CATE Predictions by Meta Learner')
    plt.xlabel('Individual Treatment Effect (ITE/CATE)')
    plt.ylabel('# of Samples')
    plt.legend(fontsize=12)
    plt.savefig(save_path)

def scatter_plot_summary(synthetic_summary, savepath, k, drop_learners=[], drop_cols=[]):
    """Generates a scatter plot comparing learner performance. Each learner's performance is plotted as a point in the
    (Abs % Error of ATE, MSE) space.
    Args:
        synthetic_summary (pd.DataFrame): summary generated by get_synthetic_summary()
        k (int): number of simulations (used only for plot title text)
        drop_learners (list, optional): list of learners (str) to omit when plotting
        drop_cols (list, optional): list of metrics (str) to omit when plotting
    """
    plot_data = synthetic_summary.drop(drop_learners).drop(drop_cols, axis=1)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    xs = plot_data["Abs % Error of ATE"]
    ys = plot_data["MSE"]

    ax.scatter(xs, ys)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    for i, txt in enumerate(plot_data.index):
        ax.annotate(
            txt,
            (
                xs[i] - np.random.binomial(1, 0.5) * xlim[1] * 0.04,
                ys[i] - ylim[1] * 0.03,
            ),
        )

    ax.set_xlabel("Abs % Error of ATE")
    ax.set_ylabel("MSE")
    ax.set_title("Learner Performance (averaged over k={} simulations)".format(k))
    plt.savefig(savepath)

def scatter_plot_summary_holdout(
    train_summary,
    validation_summary,
    k,
    savepath,
    label=["Train", "Validation"],
    drop_learners=[],
    drop_cols=[],
):
    """Generates a scatter plot comparing learner performance by training and validation.
    Args:
        train_summary (pd.DataFrame): summary for training synthetic data generated by get_synthetic_summary_holdout()
        validation_summary (pd.DataFrame): summary for validation synthetic data generated by
            get_synthetic_summary_holdout()
        label (string, optional): legend label for plot
        k (int): number of simulations (used only for plot title text)
        drop_learners (list, optional): list of learners (str) to omit when plotting
        drop_cols (list, optional): list of metrics (str) to omit when plotting
    """
    train_summary = train_summary.drop(drop_learners).drop(drop_cols, axis=1)
    validation_summary = validation_summary.drop(drop_learners).drop(drop_cols, axis=1)

    plot_data = pd.concat([train_summary, validation_summary])
    plot_data["label"] = [i.replace("Train", "") for i in plot_data.index]
    plot_data["label"] = [i.replace("Validation", "") for i in plot_data.label]

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    xs = plot_data["Abs % Error of ATE"]
    ys = plot_data["MSE"]
    group = np.array(
        [label[0]] * train_summary.shape[0] + [label[1]] * validation_summary.shape[0]
    )
    cdict = {label[0]: "red", label[1]: "blue"}

    for g in np.unique(group):
        ix = np.where(group == g)[0].tolist()
        ax.scatter(xs[ix], ys[ix], c=cdict[g], label=g, s=100)

    for i, txt in enumerate(plot_data.label[:10]):
        ax.annotate(txt, (xs[i] + 0.005, ys[i]))

    ax.set_xlabel("Abs % Error of ATE")
    ax.set_ylabel("MSE")
    ax.set_title("Learner Performance (averaged over k={} simulations)".format(k))
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))
    plt.savefig(savepath)

def bar_plot_summary_holdout(
    train_summary, validation_summary, k, savepath, drop_learners=[], drop_cols=[]
):
    """Generates a bar plot comparing learner performance by training and validation
    Args:
        train_summary (pd.DataFrame): summary for training synthetic data generated by get_synthetic_summary_holdout()
        validation_summary (pd.DataFrame): summary for validation synthetic data generated by
            get_synthetic_summary_holdout()
        k (int): number of simulations (used only for plot title text)
        drop_learners (list, optional): list of learners (str) to omit when plotting
        drop_cols (list, optional): list of metrics (str) to omit when plotting
    """
    train_summary = train_summary.drop([KEY_ACTUAL])
    train_summary["Learner"] = train_summary.index

    validation_summary = validation_summary.drop([KEY_ACTUAL])
    validation_summary["Learner"] = validation_summary.index

    for metric in ["Abs % Error of ATE", "MSE", "KL Divergence"]:
        plot_data_sub = pd.DataFrame(train_summary.Learner).reset_index(drop=True)
        plot_data_sub["train"] = train_summary[metric].values
        plot_data_sub["validation"] = validation_summary[metric].values
        plot_data_sub = plot_data_sub.set_index("Learner")
        plot_data_sub = plot_data_sub.drop(drop_learners).drop(drop_cols, axis=1)
        plot_data_sub = plot_data_sub.sort_values("train", ascending=True)

        plot_data_sub.plot(kind="bar", color=["red", "blue"], figsize=(12, 8))
        plt.xticks(rotation=30)
        plt.title(
            "Learner Performance of {} (averaged over k={} simulations)".format(
                metric, k
            )
        )
        plt.savefig(savepath)


def get_synthetic_summary_holdout(synthetic_data_func, n=1000, p = 5, valid_size=0.2, k=1):
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
    if LOAD_DATA:
        with open(f'results/experiment_2/preds_dict_train_n_{n}_p_{p}.pkl', 'rb') as handle:
            preds_dict_train = pickle.load(handle)
        with open(f'results/experiment_2/preds_dict_valid_n_{n}_p_{p}.pkl', 'rb') as handle:
            preds_dict_valid = pickle.load(handle)
        with open(f'results/experiment_2/summaries_train_n_{n}_p_{p}.pkl', 'rb') as handle:
            summaries_train = pickle.load(handle)
        with open(f'results/experiment_2/summaries_validation_n_{n}_p_{p}.pkl', 'rb') as handle:
            summaries_validation = pickle.load(handle)
    else:

        summaries_train = []
        summaries_validation = []

        for i in range(k):
            preds_dict_train, preds_dict_valid = get_synthetic_preds_holdout(
                synthetic_data_func, n=n, p = p, valid_size=valid_size
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
        
        with open(f'results/experiment_2/preds_dict_train_n_{n}_p_{p}.pkl', 'wb') as handle:
            pickle.dump(preds_dict_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'results/experiment_2/preds_dict_valid_n_{n}_p_{p}.pkl', 'wb') as handle:
            pickle.dump(preds_dict_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'results/experiment_2/summaries_train_n_{n}_p_{p}.pkl', 'wb') as handle:
            pickle.dump(summaries_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'results/experiment_2/summaries_validation_n_{n}_p_{p}.pkl', 'wb') as handle:
            pickle.dump(summaries_validation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    """
    S_learner_names = [x for x in list(preds_dict_train.keys()) if x[0]=='S']

    plot_dist(preds_dict_train, S_learner_names, f'results/experiment_2/train_dist_n_{n}_p_{p}.png')
    plot_dist(preds_dict_valid, S_learner_names, f'results/experiment_2/valid_dist_n_{n}_p_{p}.png')
    distr_plot_single_sim(preds_dict_train, f'results/experiment_2/single_sim_dist_train_n_{n}_p_{p}.png', kind='kde', linewidth=2, bw_method=0.5,
                      drop_learners=[KEY_ACTUAL] + list(preds_dict_train.keys()))
    distr_plot_single_sim(preds_dict_valid, f'results/experiment_2/single_sim_dist_valid_n_{n}_p_{p}.png', kind='kde', linewidth=2, bw_method=0.5,
                      drop_learners=[KEY_ACTUAL] + list(preds_dict_train.keys()))
    #scatter_plot_summary(preds_dict_train, 'scatterplot_dist_train.png', k = k)
    #scatter_plot_summary(preds_dict_valid, 'scatterplot_dist_valid.png', k = k)
    """
    summary_train = sum(summaries_train) / k
    summary_validation = sum(summaries_validation) / k
    
    return (
        summary_train[["Abs % Error of ATE", "MSE", "KL Divergence"]],
        summary_validation[["Abs % Error of ATE", "MSE", "KL Divergence"]],
    )

def get_synthetic_preds_holdout(
    synthetic_data_func, n=1000, p = 5, valid_size=0.2, estimators={}
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
    y, X, w, tau, b, e = synthetic_data_func(n=n, p = p)

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
        [BaseSRegressor, BaseTRegressor, BaseXRegressor],
        ["S", "T", "X"],
    ):
        for model, label_m in zip([LinearRegression, XGBRegressor, SVR,
                                   lambda: MyBaggingRegressor(SVR()),
                                   lambda: MyAdaBoostRegressor(SVR()),
                                   lambda: MyAdaBoostRegressor(MLPRegressor())], ["LR", "XGB", 'SVR',
                                                                        "Bagging_SVR","Adaboost_SVR","Adaboost_MLP"]):
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


def experiment(n,p):
    print(f"current experiment -  n: {n} p: {p}")
    train_summary, validation_summary = get_synthetic_summary_holdout(
        simulate_nuisance_and_easy_treatment,
        n = n,
        p = p,
        valid_size = 0.2,
        k = 1
    )

    """
    scatter_plot_summary_holdout(train_summary,
                            validation_summary,
                            k=10,
                            label=['Train', 'Validation'],
                            drop_learners=[],
                            drop_cols=[])
                            """
    #print(train_summary)
    #print(validation_summary)
    train_summary.to_csv(f'results/experiment_2/train_summary_p_{p}_n_{n}.csv')
    validation_summary.to_csv(f'results/experiment_2/validation_summary_p_{p}_n_{n}.csv')
    print(f"Finished experiment -  n: {n} p: {p}")

import multiprocessing

if __name__ == '__main__':
    p_values = [5, 10, 100, 200]
    n_values = [200, 500, 1000, 10_000, 50_000]
    
    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(experiment, [(n,p) for n in n_values for p in p_values])
            
