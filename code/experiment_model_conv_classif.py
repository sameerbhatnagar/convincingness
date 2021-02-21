import os
import json
import spacy
import pandas as pd
import plac
import numpy as np
from itertools import compress

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

from plots_data_loaders import load_data
from data_loaders import BASE_DIR
from feature_extraction_features_by_discipline import (
    NUMERIC_COLUMNS,
    BINARY_COLUMNS,
    TARGETS,
)

nlp = spacy.load("en_core_web_md")

CONV_THRESH_RATIO = 0.75
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]


def white_space_analyzer(text):
    words = text.split(" ")
    for w in words:
        yield w


def append_pos_rationale_representations(df):
    """
    """
    d = list(
        zip(
            df["id"].to_list(),
            [
                " ".join([token.pos_ for token in doc])
                for doc in nlp.pipe(df["rationale"])
            ],
            [
                " ".join(
                    [
                        token.lemma_.lower()
                        for token in doc
                        if not token.is_stop and token.is_alpha
                    ]
                )
                for doc in nlp.pipe(df["rationale"])
            ],
            [
                " ".join(
                    [
                        f"{chunk.root.text}-{chunk.root.dep_}-{chunk.root.head.text}"
                        for chunk in doc.noun_chunks
                    ]
                )
                for doc in nlp.pipe(df["rationale"])
            ],
            [
                " ".join(
                    [
                        f"{token.lemma_.lower()}-{token.pos_}"
                        for token in doc
                        if not token.is_stop
                    ]
                )
                for doc in nlp.pipe(df["rationale"])
            ],
        )
    )
    df = pd.merge(
        df,
        pd.DataFrame(
            d,
            columns=[
                "id",
                "rationale_pos",
                "rationale_content_words",
                "rationale_chunk_dependencies",
                "rationale_text_pos",
            ],
        ),
        on="id",
    )
    return df


def get_important_feature_names(pipe, clf, numeric_columns, X_test, y_test):
    """
    """
    feature_names = (
        numeric_columns
        #         +pipe[0].named_transformers_["one_hot"].get_feature_names().tolist()
        + pipe[0].named_transformers_["ngram"].get_feature_names()
        + pipe[0].named_transformers_["pos"].get_feature_names()
        + pipe[0].named_transformers_["chunk_dep"].get_feature_names()
        + pipe[0].named_transformers_["text_pos"].get_feature_names()
    )
    feature_types = [n.split("_")[0] for n in numeric_columns]
    #     feature_types.extend(
    #         ["transition" for i in range(len(pipe[0].named_transformers_["one_hot"].get_feature_names()))]
    #     )
    feature_types.extend(
        [
            "ngram"
            for i in range(
                len(pipe[0].named_transformers_["ngram"].get_feature_names())
            )
        ]
    )
    feature_types.extend(
        [
            "pos"
            for i in range(len(pipe[0].named_transformers_["pos"].get_feature_names()))
        ]
    )
    feature_types.extend(
        [
            "chunk_dep"
            for i in range(
                len(pipe[0].named_transformers_["chunk_dep"].get_feature_names())
            )
        ]
    )
    feature_types.extend(
        [
            "text_pos"
            for i in range(
                len(pipe[0].named_transformers_["text_pos"].get_feature_names())
            )
        ]
    )
    feature_names = list(
        compress(
            [f"{ft}_{fn}" for ft, fn in zip(feature_types, feature_names)],
            pipe.named_steps["variancethreshold"].get_support(),
        )
    )

    feature_names = list(
        compress(feature_names, pipe.named_steps["selectkbest"].get_support(),)
    )

    r = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=0)
    f = []
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            fd = {}
            fd["name"] = feature_names[i]
            fd["importance"] = r.importances_mean[i].round(3)
            fd["importance_std"] = r.importances_std[i].round(3)
            f.append(fd)
    return f


def main(
    discipline: (
        "Discipline",
        "positional",
        None,
        str,
        ["Physics", "Ethics", "Chemistry", "UKP", "IBM_ArgQ", "IBM_Evi"],
    ),
    output_dir_name: ("Directory name for results", "positional", None, str,),
    task: (
        "Regression or Classification",
        "positional",
        None,
        str,
        ["regress", "classif"],
    ),
    target: (
        "Which target tto regress to",
        "positional",
        None,
        str,
        ["all","y_winrate_nopairs","y_winrate","y_BT","y_elo","y_crowdBT"],
    ),
    long_only: (
        "focus only on long explanations",
        "flag", "l", bool,
    ),
):

    df = load_data(discipline, output_dir_name)

    topics = df["topic"].value_counts().index.tolist()
    if target == "all":
        targets = TARGETS[discipline]
    else:
        targets = (target,)

    if task == "classif":
        # label top quartile as positive class for convincingness
        for target in targets:
            for topic in topics:
                conv_thresh_ratio = np.quantile(
                    df[df["topic"] == topic][target], q=CONV_THRESH_RATIO
                )
                df.loc[
                    ((df[target] >= conv_thresh_ratio) & (df["topic"] == topic)),
                    f"{target}_bin",
                ] = 1
                df.loc[
                    ((df[target] < conv_thresh_ratio) & (df["topic"] == topic)),
                    f"{target}_bin",
                ] = 0

        clfs = [
            ("length", None),
            ("log_reg", LogisticRegression()),
            ("svc", SVC()),
            ("dtree", DecisionTreeClassifier(max_depth=5)),
            ("rf", RandomForestClassifier(max_depth=5)),
        ]

        targets = [f"{t}_bin" for t in targets]
    elif task == "regress":
        clfs = [
            ("length", None),
            ("lin_reg", LinearRegression()),
            # ("svr",SVR()),
            ("dtree", DecisionTreeRegressor(max_depth=5)),
            ("rf", RandomForestRegressor(max_depth=5)),
        ]

    # calculate POS and other representations of rationale text
    print(f"calculate POS and other representations of rationale text")
    df = append_pos_rationale_representations(df)

    numeric_columns = NUMERIC_COLUMNS[discipline]
    binary_columns = BINARY_COLUMNS[discipline]

    features_pipe = ColumnTransformer(
        [
            ("scale", StandardScaler(), numeric_columns),
            ("ngram", TfidfVectorizer(use_idf=True,), "rationale_content_words"),
            ("pos", TfidfVectorizer(use_idf=False), "rationale_pos"),
            (
                "chunk_dep",
                TfidfVectorizer(use_idf=False, analyzer=white_space_analyzer),
                "rationale_chunk_dependencies",
            ),
            (
                "text_pos",
                TfidfVectorizer(use_idf=False, analyzer=white_space_analyzer),
                "rationale_text_pos",
            ),
        ]
    )
    if task == "classif":
        features_pipe_full = make_pipeline(
            features_pipe,
            VarianceThreshold(),
            SelectKBest(score_func=f_classif, k=768),
        )
    elif task == "regress":
        features_pipe_full = make_pipeline(
            features_pipe,
            VarianceThreshold(),
            SelectKBest(score_func=f_regression, k=768),
        )

    results = []
    if long_only:
        quartiles = QUARTILES[-1:]
    else:
        quartiles = QUARTILES

    for target in targets:
        print(target)
        drop_cols = ["topic", "wc_bin", target]
        all_columns = (
            numeric_columns
            + binary_columns
            + drop_cols
            + [
                "rationale_content_words",
                "rationale_pos",
                "rationale_chunk_dependencies",
                "rationale_text_pos",
            ]
        )
        df2 = df[all_columns + ["id"]]  # .dropna()

        for t, topic in enumerate(topics):
            df_train = df2[
                (df2["topic"] != topic) & (df2["wc_bin"].isin(quartiles))
            ].dropna(subset=[target])
            y_train = df_train[target]
            X_train_df = df_train.drop(drop_cols + ["id"], axis=1)

            df_test = df2[
                (df2["topic"] == topic) & (df2["wc_bin"].isin(quartiles))
            ].dropna(subset=[target])
            y_test = df_test[target].fillna(df_test[target].mean())
            X_test_df = df_test.drop(drop_cols + ["id"], axis=1)
            X_test_df_lookup = df_test.drop(drop_cols, axis=1)

            print(f"\t{t} : {topic}")
            print(f"\t train: {X_train_df.shape}; test: {X_test_df.shape}")

            for clf_name, clf in clfs:
                print(f"\t\t{clf_name}")
                if clf_name == "length":
                    if task == "classif":
                        df_train = (
                            df[(df["topic"] != topic) & (df["wc_bin"].isin(quartiles))]
                            .dropna(subset=[target])["rationale"]
                            .str.count("\w+")
                            .values.reshape(-1, 1)
                        )
                        clf = LogisticRegression()
                        clf.fit(df_train, y_train)
                        y_pred = clf.predict(
                            X_test_df["rationale_content_words"]
                            .str.count("\w+")
                            .values.reshape(-1, 1)
                        )
                    elif task == "regress":
                        y_pred = df_test["rationale_content_words"].str.count("\w+")

                    important_features = {"name": "length", "importance": 1}
                else:
                    features_pipe_full.fit(X_train_df, y_train)
                    X_train = features_pipe_full.transform(X_train_df)
                    clf.fit(X_train, y_train)
                    X_test = features_pipe_full.transform(X_test_df)
                    y_pred = clf.predict(X_test)

                    important_features = get_important_feature_names(
                        features_pipe_full,
                        clf,
                        numeric_columns,
                        X_test.toarray(),
                        y_test,
                    )

                d = {}
                d["target"] = target
                d["topic"] = topic
                d["clf"] = clf_name
                if task == "classif":
                    d["f1"] = f1_score(y_true=y_test, y_pred=y_pred)
                else:
                    d["pearsonr"] = pearsonr(y_pred, y_test)[0]
                    d["spearmanr"] = spearmanr(y_pred, y_test)[0]
                    d["rank_scores"] = [
                        {"arg_id": int(i), "rank_score": float(p)}
                        for i, p in zip(
                            df_test
                            .dropna(subset=[target])["id"]
                            .values,
                            y_pred,
                        )
                    ]
                d["N"] = X_test_df.shape[0]
                d["features"] = important_features
                results.append(d)

            population = "switchers"
            if long_only:
                fp = os.path.join(
                    BASE_DIR,
                    "tmp",
                    output_dir_name,
                    discipline,
                    population,
                    f"pred_rank_score_{task}_{target}_{discipline}_{population}_longest.json",
                )
            else:
                fp = os.path.join(
                    BASE_DIR,
                    "tmp",
                    output_dir_name,
                    discipline,
                    population,
                    f"pred_rank_score_{task}_{target}_{discipline}_{population}.json",
                )
            with open(fp, "w") as f:
                json.dump(results, f,indent=2)


if __name__ == "__main__":
    plac.call(main)
