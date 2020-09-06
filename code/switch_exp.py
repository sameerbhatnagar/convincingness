import os
import json
import plac
from itertools import compress
from joblib import dump

import warnings

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", HessianInversionWarning)


from features_shown import get_feature_names, append_features_shown

import pandas as pd
import numpy as np
import data_loaders
from feature_extraction import append_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, Normalizer
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold, SelectKBest

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from argBT import RESULTS_DIR

MIN_TRAINING_RECORDS = 20


def get_pipeline(feature_columns_numeric, feature_columns_categorical, df):

    n_quantiles = int(0.75 * df.shape[0])
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("quantile_transformer", QuantileTransformer(n_quantiles=n_quantiles)),
        ]
        # steps=[("imputer", SimpleImputer()), ("std_scaler", StandardScaler()),]
    )

    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])

    feature_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, feature_columns_numeric),
            ("categorical", categorical_transformer, feature_columns_categorical),
        ]
    )

    feature_transform_select = Pipeline(
        steps=[
            ("feature_transformer", feature_transformer),
            ("var_thresh", VarianceThreshold()),
            # ("select_k_best",SelectKBest(k=5))
        ]
    )

    return feature_transform_select


def main_by_topic(df_train, kwargs):

    feature_columns_numeric = kwargs.get("feature_columns_numeric")
    feature_columns_categorical = kwargs.get("feature_columns_categorical")
    target = kwargs.get("target")
    results_dir_discipline = kwargs.get("results_dir_discipline")
    topic = kwargs.get("topic")
    timestep = kwargs.get("timestep")

    # only make predictions for students, not teacher answers, where user token is ""/null
    feature_names = feature_columns_numeric + feature_columns_categorical
    df_train_data = df_train.loc[~(df_train["user_token"].isna()), feature_names]

    y = df_train.loc[~(df_train["user_token"].isna()), target].values

    feature_transformer = get_pipeline(
        feature_columns_numeric=feature_columns_numeric,
        feature_columns_categorical=feature_columns_categorical,
        df=df_train,
    )

    # run feature transformation on all rows
    X = feature_transformer.fit_transform(df_train_data, y)

    # names for variables in statsmodels
    feature_names = list(
        compress(
            feature_names, feature_transformer.named_steps["var_thresh"].get_support(),
        )
    )
    # feature_names=compress(
    #     feature_names,
    #     feature_transformer.named_steps["select_k_best"].get_support()
    # )

    # rebuild dataframe, but with sklearn feature transformation (for statsmodels)
    X_df = pd.DataFrame(X, columns=feature_names,)

    d = {}
    for name, clf in [
        ("LR", LogisticRegression()),
        ("RF", RandomForestClassifier(max_depth=3)),
    ]:

        # train model on all but last row
        X_train = X[:-1, :]
        if X_train.shape[0] < MIN_TRAINING_RECORDS:
            return {}
            
        clf.fit(X_train, y[:-1])

        # make prediction on last row
        X_test = X[-1, :].reshape(1, -1)
        d.update({
            "prediction_{}".format(name): list(clf.predict(X_test))
        })

        if name == "LR":
            weights = clf.coef_[0]
        elif name == "RF":
            weights = clf.feature_importances_
        d.update({
            "feature_names_{}".format(name): list(zip(weights, feature_names)),
        })

    logit_model = sm.Logit(y, sm.add_constant(X_df))
    logit_model_results = logit_model.fit(method="bfgs", disp=0)
    d.update({
        "topic": topic,
        "timestep": timestep,
        "n": X_df.shape[0],
        "r2": np.round(logit_model_results.prsquared, 2),
        "params": {
            k: np.round(v, 3)
            for k, v in logit_model_results.params[logit_model_results.pvalues < 0.05]
            .to_dict()
            .items()
        },
        "params_all": {
            k: np.round(v, 3) for k, v in logit_model_results.params.to_dict().items()
        },
        "transition":df_train.tail(1)["transition"].iat[0],
        "test_answer_id":str(df_train.tail(1)["id"].iat[0]),
        "y_true":df_train.tail(1)[target].iat[0],
    })
    return d


def main(discipline, feature_types_included="all"):
    """
    Command line utility to run experiments on predicting
    whether a student will choose a peer's explanation
    over their own in Technology Mediated Peer Instruction

    Arguments:
    =========
        - discipline -> str
            - options: Physics, Chemistry, Biology,
            Statistics, or Ethics
        - feature_types_included -> str
            - options: all, convincingness, surface
    Returns:
    ========
        None
    """
    print(discipline)

    data_dir_discipline = os.path.join(RESULTS_DIR, discipline, "data")

    # sort files by size to get biggest ones done first
    # https://stackoverflow.com/a/20253803
    all_files = (
        os.path.join(data_dir_discipline, filename)
        for basedir, dirs, files in os.walk(data_dir_discipline)
        for filename in files
    )
    all_files = sorted(all_files, key=os.path.getsize)  # , reverse=True)

    topics = [os.path.basename(fp)[:-4] for fp in all_files]

    if feature_types_included == "all":
        feature_types_included = [
            "surface",
            "convincingness",
        ]
        results_dir_discipline = os.path.join(RESULTS_DIR, discipline, "results")
    else:
        results_dir_discipline = os.path.join(
            RESULTS_DIR, discipline, "results_{}".format(feature_types_included)
        )
        feature_types_included = [feature_types_included]

    # make list of what has not already been done
    topics_already_done = [fp[:-5] for fp in os.listdir(results_dir_discipline)]
    topics_to_do = [t for t in topics if t not in topics_already_done]

    feature_columns_numeric, feature_columns_categorical = get_feature_names(
        feature_types_included=feature_types_included, discipline=discipline
    )

    target = "switch_exp"

    kwargs = {
        "feature_columns_categorical": feature_columns_categorical,
        "target": target,
        "LSA_features": False,
        "feature_types_included": feature_types_included,
    }

    for t, topic in enumerate(topics_to_do):

        print("{}/{}: {}".format(t, len(topics_to_do), topic))

        # read answer data
        df = pd.read_csv(os.path.join(data_dir_discipline, "{}.csv".format(topic)),)
        df["rationale"] = df["rationale"].fillna("")

        results = []
        # run training on all previous times steps; test on current time step
        for counter, (r, _) in enumerate(df.groupby("a_rank_by_time")):

            if r % 10 == 0:
                print(
                    "\t\ttime step {}/{}".format(
                        counter, df["a_rank_by_time"].value_counts().shape[0],
                    )
                )

            df_train = df[df["a_rank_by_time"] <= r].copy()
            if df_train.shape[0] > MIN_TRAINING_RECORDS:
                df_train = append_features(
                    df=df_train,
                    feature_types_included=feature_types_included,
                    timestep=r,
                )

                # derive features based on shown rationales, for feature_types_included
                df_train = append_features_shown(df_train, kwargs)

                # some columns don't even come back from append_features_shown function
                # the convincingness features of a rationale should not be
                # used in predicting switch_exp, as it is not known yet
                feature_columns_numeric = [
                    f
                    for f in feature_columns_numeric
                    if f in df_train.columns
                    and f not in ["convincingness_BT", "convincingness_baseline"]
                ]
                # filter out columns that are all NA
                feature_columns_numeric_non_null = [
                    f
                    for f, fn in zip(
                        feature_columns_numeric,
                        df_train[feature_columns_numeric].isnull().values.all(axis=0),
                    )
                    if not fn
                ]

                if (
                    len(feature_columns_numeric_non_null) > 0
                    and df_train.shape[0] > MIN_TRAINING_RECORDS
                ):
                    kwargs.update(
                        {
                            "topic": topic,
                            "results_dir_discipline": results_dir_discipline,
                            "feature_columns_numeric": feature_columns_numeric_non_null,
                            "timestep": r,
                        }
                    )
                    df_train = df_train.sort_values("a_rank_by_time")
                    d = main_by_topic(df_train, kwargs)
                    results.append(d)
                else:
                    print("skipping topic")

        fname = os.path.join(results_dir_discipline, "{}.json".format(topic))
        with open(fname, "w") as f:
            json.dump(results,f, indent=2)


if __name__ == "__main__":
    import plac

    plac.call(main)
