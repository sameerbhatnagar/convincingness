import os
import json
import plac

from make_corpus import get_mydalite_answers
from utils import get_vocab

import pandas as pd
import numpy as np
import data_loaders

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

from feature_extraction import PRE_CALCULATED_FEATURES

RESULTS_DIR = os.path.join(data_loaders.BASE_DIR, "tmp", "switch_exp")
MIN_WORD_COUNT = 5


def get_pipeline(feature_columns_numeric, feature_columns_categorical):

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()), ("std_scaler", StandardScaler()),]
    )

    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])

    feature_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, feature_columns_numeric),
            ("categorical", categorical_transformer, feature_columns_categorical),
        ]
    )

    return feature_transformer


def shown_feature_counts(shown_ids_str, df, feature):
    avg_feature_value = df[feature].mean()
    shown_ids = shown_ids_str.strip("[]").replace(" ", "").split(",")
    shown_wc = []
    for k in shown_ids:
        if k != "":
            try:
                shown_wc.append(
                    df.loc[df["id"]==int(k), feature].iat[0] if not (df.loc[df["id"]==int(k), feature].isna().iat[0]) else avg_feature_value
                )
            except IndexError:  # data got filtered out
                shown_wc.append(avg_feature_value)
    return shown_wc

def append_features_shown_LSA(df_q):
    """
    Arguments:
    =========
        df_q: all answers for a particular question, with "rationale" and shown
                "rationales" columns. Ordered by "a_rank_by_time" (i.e.
                when the answer was put in DB)
    Returns:
    =======
        df_q : add columns for <mean, max> LSA distances to
                - shown "rationales", all other rationales
    """
    # make LSA space with all rationales
    corpus = df_q["rationale"]
    vocab = get_vocab(corpus)
    n_components = 100 if len(vocab) > 100 else len(vocab) - 1
    lsa = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("svd", TruncatedSVD(n_components=n_components)),
            ("norm", Normalizer()),
        ]
    )

    lsa.fit(corpus)

    # iterate through each student answer
    for rank in df_q["a_rank_by_time"].values[5:]:

        # look up vector for each shown answer and append cos sim
        if len(df_q[df_q["a_rank_by_time"] == rank].rationales) > 0:
            shown_ids = [
                int(s)
                for s in df_q[df_q["a_rank_by_time"] == rank]
                .rationales.iat[0]
                .strip("[]")
                .replace(" ", "")
                .split(",")
            ]

            try:
                df_q.loc[
                    df_q["a_rank_by_time"] == rank, "mean_cosine_sim_to_shown"
                ] = cosine_similarity(
                    lsa.transform(df_q[df_q["a_rank_by_time"] == rank].rationale),
                    lsa.transform(df_q.loc[shown_ids, "rationale"]),
                ).mean()
            except KeyError:
                pass  #

        # add mean cosine similarity to all rationales
        df_q.loc[
            df_q["a_rank_by_time"] == rank, "mean_cosine_sim_to_others"
        ] = cosine_similarity(
            lsa.transform(df_q[df_q["a_rank_by_time"] == rank].rationale),
            lsa.transform(df_q["rationale"]),
        ).mean()



def append_features_shown(df,kwargs):
    """
    for each feature in  pre_calculated_features,
    for each answer, aggregate feature values for those rationales that were shown
    Same for LSA vectors
    """

    LSA_features = kwargs.get("LSA_features")
    pre_calculated_features = kwargs.get("pre_calculated_features")

    for feature in pre_calculated_features:
        feature_name = "shown_{}".format(feature)
        df[feature_name] = df["rationales"].apply(
            shown_feature_counts, args=(df, feature,)
        )

        # for the answers where we have "shown rationale data"
        # take the <max,min,mean> of shown <feature> as feature
        df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), feature_name + "_mean"
        ] = df.loc[df[feature_name].apply(lambda x: len(x) > 0), feature_name].apply(
            lambda x: np.array(x).mean()
        )

        df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), feature_name + "_max"
        ] = df.loc[df[feature_name].apply(lambda x: len(x) > 0), feature_name].apply(
            lambda x: np.array(x).max()
        )

        df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), feature_name + "_min"
        ] = df.loc[df[feature_name].apply(lambda x: len(x) > 0), feature_name].apply(
            lambda x: np.array(x).min()
        )

        if feature == "rationale_word_count":
            # take the number of shown rationales with word_count<MIN_WORD_COUNT
            # as feature
            df.loc[
                df[feature_name].apply(lambda x: len(x) > 0), "n_shown_short"
            ] = df.loc[
                df[feature_name].apply(lambda x: len(x) > 0), feature_name
            ].apply(
                lambda x: len([s for s in x if s < MIN_WORD_COUNT])
            )

            # take the number of shown rationales with word_count less than 0.5 times
            # their own rationale_word_count as feature
            df.loc[
                df[feature_name].apply(lambda x: len(x) > 0), "n_shown_shorter_than_own"
            ] = df.loc[
                df[feature_name].apply(lambda x: len(x) > 0), [feature, feature_name]
            ].apply(
                lambda x: len([i for i in x[feature_name] if i < 0.5 * x[feature]]),
                axis=1,
            )

            # take the number of shown rationales with word_count greater than 1.5 times
            # their own rationale_word_count as feature
            df.loc[
                df[feature_name].apply(lambda x: len(x) > 0), "n_shown_longer_than_own"
            ] = df.loc[
                df[feature_name].apply(lambda x: len(x) > 0), [feature, feature_name]
            ].apply(
                lambda x: len([i for i in x[feature_name] if i > 1.5 * x[feature]]),
                axis=1,
            )

    # features based on cosine similarity of LSA vectors
    # (TfIdf + SVD + Norm)
    df_q = df.sort_values("a_rank_by_time")
    df_q["a_rank_by_time"] = list(range(df_q.shape[0]))

    if LSA_features:
        df_q = append_features_shown_LSA(df_q)

    return df_q


def main_by_topic(df_q, kwargs):

    feature_columns_numeric = kwargs.get("feature_columns_numeric")
    feature_columns_categorical = kwargs.get("feature_columns_categorical")
    target = kwargs.get("target")
    results_dir_discipline = kwargs.get("results_dir_discipline")
    topic = kwargs.get("topic")

    feature_transformer = get_pipeline(
        feature_columns_numeric=feature_columns_numeric,
        feature_columns_categorical=feature_columns_categorical,
    )

    # only make predictions for students, not teacher answers, where user token is ""/null
    # for Wrong--> Right transitions

    X = feature_transformer.fit_transform(
        df_q.loc[:,
            feature_columns_numeric + feature_columns_categorical,
        ]
    )
    y = df_q[target].values  # .reshape(-1,1)

    scores = cross_val_score(LogisticRegression(), X, y)
    scores_f1 = cross_val_score(LogisticRegression(), X, y, scoring="f1")

    # names for variables in statsmodels
    X_df = pd.DataFrame(
        X,
        columns= feature_columns_numeric + list(
            feature_transformer.named_transformers_["categorical"]
            .steps[0][1]
            .get_feature_names(feature_columns_categorical)
        ),
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
            "acc": np.round(np.mean(scores), 2),
            "sd": np.round(np.std(scores), 2),
            "f1": np.round(np.mean(scores_f1), 2),
            "f1_sd": np.round(np.std(scores_f1), 2),
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
            }
        }
        fname = os.path.join(results_dir_discipline, "{}.json".format(topic))
        with open(fname, "w") as f:
            f.write(json.dumps(d, indent=2))
    except np.linalg.LinAlgError as e:
        print(e)
        pass

    return


def main(discipline):
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

    results_dir_discipline = os.path.join(RESULTS_DIR, discipline, "results")
    topics_already_done = [fp[:-5] for fp in os.listdir(results_dir_discipline)]

    topics_to_do = [t for t in topics if t not in topics_already_done]

    feature_types_included = [
        "convincingness"
    ]

    pre_calculated_features = [
        x for x in [
            PRE_CALCULATED_FEATURES[feature_type] for feature_type in feature_types_included]
            for x in x
        ]
    feature_columns_numeric = (
        [
            "a_rank_by_time",
            "rationale_word_count",
            # "n_shown_short",
            # "n_shown_shorter_than_own",
            # "n_shown_longer_than_own",
            # "P_switch_exp_student",
            # "P_switch_exp_topic",
            # "mean_cosine_sim_to_shown",
            # "mean_cosine_sim_to_others",
            # "q_diff1",
            "student_strength1",
        ]
        + pre_calculated_features
        + ["shown_{}_mean".format(feature) for feature in pre_calculated_features]
        + ["shown_{}_max".format(feature) for feature in pre_calculated_features]
        + ["shown_{}_min".format(feature) for feature in pre_calculated_features]
    )
    feature_columns_categorical = [
        "first_correct",
    ]
    target = "switch_exp"

    kwargs = {
        "feature_columns_categorical": feature_columns_categorical,
        "target": target,
        "LSA_features": False,
        "pre_calculated_features": pre_calculated_features,
    }


    for t, topic in enumerate(topics_to_do):

        print("{}/{}: {}".format(t, len(topics_to_do), topic))

        # read answer data
        df = pd.read_csv(
            os.path.join(data_dir_discipline, "{}.csv".format(topic)), index_col="id"
        )
        df["rationale"] = df["rationale"].fillna("")

        # append columns with pre-calculated features
        features_dir = os.path.join(data_dir_discipline+"_with_features",topic+"_features")
        for feature_type in feature_types_included:

            feature_type_fpath = os.path.join(
                features_dir, topic+"_"+feature_type + ".json"
            )
            with open(feature_type_fpath,"r") as f:
                features_json = json.load(f)

            # each feature_type includes multiple features
            for feature in features_json:
                df = pd.merge(
                    df,
                    pd.DataFrame(features_json[feature], columns=["id",feature]),
                    on="id",
                    how="left"
                )

        # derive features based on shown rationales
        df_q = append_features_shown(df,kwargs)

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

        df_q_w = df_q[(
            (df_q["user_token"].isna() == False)
            &
            (df_q["transition"].isin(["wr","ww"]))
            )]

        if len(feature_columns_numeric_non_null) > 0 and df_q_w.shape[0]>10:
            kwargs.update(
                {
                    "topic": topic,
                    "results_dir_discipline": results_dir_discipline,
                    "feature_columns_numeric": feature_columns_numeric_non_null,
                }
            )
            main_by_topic(df_q_w, kwargs)
        else:
            print("skipping topic")


if __name__ == "__main__":
    import plac

    plac.call(main)
