import os
import json
import plac
from joblib import dump


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
import statsmodels.api as sm


RESULTS_DIR = os.path.join(data_loaders.BASE_DIR, "tmp", "switch_exp")


def get_pipeline(feature_columns_numeric, feature_columns_categorical, df):

    n_quantiles = int(0.75*df.shape[0])
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()), ("quantile_transformer", QuantileTransformer(n_quantiles=n_quantiles)),]
        # steps=[("imputer", SimpleImputer()), ("std_scaler", StandardScaler()),]
    )

    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])

    feature_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, feature_columns_numeric),
            ("categorical", categorical_transformer, feature_columns_categorical),
        ]
    )

    return feature_transformer



def main_by_topic(df_q, kwargs):

    feature_columns_numeric = kwargs.get("feature_columns_numeric")
    feature_columns_categorical = kwargs.get("feature_columns_categorical")
    target = kwargs.get("target")
    results_dir_discipline = kwargs.get("results_dir_discipline")
    topic = kwargs.get("topic")

    feature_transformer = get_pipeline(
        feature_columns_numeric=feature_columns_numeric,
        feature_columns_categorical=feature_columns_categorical,
        df = df_q,
    )

    # only make predictions for students, not teacher answers, where user token is ""/null
    feature_names = feature_columns_numeric + feature_columns_categorical
    df_q_data = df_q.loc[~(df_q["user_token"].isna()), feature_names]
    y = df_q.loc[~(df_q["user_token"].isna()),target].values  # .reshape(-1,1)

    scores_acc = {}
    scores_acc_sd = {}
    scores_f1 = {}
    scores_f1_sd = {}

    results_dir_discipline_models_all = os.path.join(results_dir_discipline, "models")
    if not os.path.exists(results_dir_discipline_models_all):
        os.mkdir(results_dir_discipline_models_all)

    for name, clf in [
        ("LR", LogisticRegression()),
        ("SVM", SVC()),
        ("RF", RandomForestClassifier(max_depth=5)),
    ]:
        clf = Pipeline([
        ("feature_transformer",feature_transformer),
        ("classifier",clf)
        ])

        cv_scores = cross_val_score(clf, df_q_data, y)
        scores_acc[name] = np.mean(cv_scores)
        scores_acc_sd[name] = np.std(cv_scores)

        cv_scores = cross_val_score(clf, df_q_data, y, scoring="f1")
        scores_f1[name] = np.mean(cv_scores)
        scores_f1_sd[name] = np.std(cv_scores)

        clf.fit(df_q_data, y)

        results_dir_discipline_model = os.path.join(results_dir_discipline_models_all, name)
        if not os.path.exists(results_dir_discipline_model):
            os.mkdir(results_dir_discipline_model)

        fname = os.path.join(
            results_dir_discipline_model, "{}_{}.pkl".format(name, topic)
        )

        with open(fname, "wb") as f:
            dump(clf, f)

        fname = os.path.join(
            results_dir_discipline_model, "{}_features_{}.json".format(name, topic)
        )
        with open(fname,"w") as f:
            json.dump(feature_names,f)

    # names for variables in statsmodels
    if len(feature_columns_categorical)>0:
        feature_names = feature_columns_numeric + list(
            feature_transformer.named_transformers_["categorical"]
            .steps[0][1]
            .get_feature_names(feature_columns_categorical)
        )
    else:
        feature_names = feature_columns_numeric

    # rebuild dataframe, but with sklearn feature transformation
    X_df = pd.DataFrame(
        feature_transformer.fit_transform(df_q_data),
        columns=feature_names,
    )

    drop_cols = X_df.columns[X_df.sum() == 0]
    X_df = X_df.drop(drop_cols, axis=1)

    logit_model = sm.Logit(y, sm.add_constant(X_df))
    try:
        logit_model_results = logit_model.fit()
        d = {
            "topic": topic,
            "n": df_q.shape[0],
            "baseline": np.round(df_q[target].value_counts(normalize=True).max(), 2),
            "acc": scores_acc,
            "acc_sd": scores_acc_sd,
            "f1": scores_f1,
            "f1_sd": scores_f1_sd,
            "r2": np.round(logit_model_results.prsquared, 2),
            "dropped_cols": list(drop_cols),
            "params": {
                k: np.round(v, 3)
                for k, v in logit_model_results.params[
                    logit_model_results.pvalues < 0.01
                ]
                .to_dict()
                .items()
            },
            "params_all": {
                k: np.round(v, 3)
                for k, v in logit_model_results.params.to_dict().items()
            },
        }
        fname = os.path.join(results_dir_discipline, "{}.json".format(topic))
        with open(fname, "w") as f:
            f.write(json.dumps(d, indent=2))

    except np.linalg.LinAlgError as e:
        print(e)
        pass

    return


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
    all_files = sorted(all_files, key=os.path.getsize, reverse=True)

    topics = [os.path.basename(fp)[:-4] for fp in all_files]

    if feature_types_included == "all":
        feature_types_included = [
            "surface",
            "convincingness",
        ]
        results_dir_discipline = os.path.join(RESULTS_DIR, discipline, "results")
    else:
        results_dir_discipline = os.path.join(RESULTS_DIR, discipline, "results_{}".format(feature_types_included))
        feature_types_included=[feature_types_included]

    # make directory for results if it doesn't already exits
    if not os.path.exists(results_dir_discipline):
        os.mkdir(results_dir_discipline)

    # make list of what has not already been done
    topics_already_done = [fp[:-5] for fp in os.listdir(results_dir_discipline)]
    topics_to_do = [t for t in topics if t not in topics_already_done]


    feature_columns_numeric,feature_columns_categorical = get_feature_names(
        feature_types_included=feature_types_included,
        discipline=discipline
    )

    target = "switch_exp"

    kwargs = {
        "feature_columns_categorical": feature_columns_categorical,
        "target": target,
        "LSA_features": False,
        "feature_types_included":feature_types_included,
    }

    for t, topic in enumerate(topics_to_do):

        print("{}/{}: {}".format(t, len(topics_to_do), topic))

        # read answer data
        df = pd.read_csv(
            os.path.join(data_dir_discipline, "{}.csv".format(topic)), index_col="id"
        )
        df["rationale"] = df["rationale"].fillna("")

        df = append_features(df)

        # derive features based on shown rationales, for feature_types_included
        df_q = append_features_shown(df, kwargs)

        # some columns don't even come back from append_features_shown function
        feature_columns_numeric = [
            f for f in feature_columns_numeric if f in df_q.columns
        ]
        # filter out columns that are all NA
        feature_columns_numeric_non_null = [
            f
            for f, fn in zip(
                feature_columns_numeric,
                df_q[feature_columns_numeric].isnull().values.all(axis=0),
            )
            if not fn
        ]

        # df_q_w = df_q[(
        #     (df_q["user_token"].isna() == False)
        #     &
        #     (df_q["transition"].isin(["wr","ww"]))
        #     )]

        if len(feature_columns_numeric_non_null) > 0 and df_q.shape[0] > 10:
            kwargs.update(
                {
                    "topic": topic,
                    "results_dir_discipline": results_dir_discipline,
                    "feature_columns_numeric": feature_columns_numeric_non_null,
                }
            )
            main_by_topic(df_q, kwargs)
        else:
            print("skipping topic")


if __name__ == "__main__":
    import plac

    plac.call(main)
