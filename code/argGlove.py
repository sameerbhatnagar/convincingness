import data_loaders
import utils
import pandas as pd
import time
import numpy as np
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm
from sklearn.metrics import accuracy_score

import spacy

NLP_MD = spacy.load("en_core_web_md", disable=["ner", "tagger", "parser"])


class GloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._nlp = NLP_MD

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.concatenate([doc.vector.reshape(1, -1) for doc in self._nlp.pipe(X)])


CROSS_TOPIC = True
data = {}
data_sources = ["IBM_ArgQ", "UKP"]
for data_source in data_sources:
    data[data_source] = data_loaders.load_arg_pairs(
        data_source, cross_topic_validation=CROSS_TOPIC
    )

data["dalite"] = data_loaders.load_dalite_data(cross_topic_validation=CROSS_TOPIC)
data["IBM_Evi"] = data_loaders.load_arg_pairs_IBM_Evi(
    cross_topic_validation=CROSS_TOPIC
)

total_t0 = time.time()

import sys

data_source = sys.argv[1]

if data_source == "all":
    data_sources = data.keys()
else:
    data_sources = [data_source]

df_results_all = pd.DataFrame()
for data_source in data_sources:
    print(data_source)
    train_dataframes, test_dataframes, df_all = data[data_source]
    df_all = df_all.rename(columns={"question": "topic"})

    t_source = time.time()

    df_test_all = pd.DataFrame()

    for i, (df_train, df_test) in enumerate(zip(train_dataframes, test_dataframes)):
        print("Fold {}".format(i))
        t_fold = time.time()

        df_train["y"] = df_train["label"].map({"a1": -1, "a2": 1})
        df_test["y"] = df_test["label"].map({"a1": -1, "a2": 1})
        df_train = df_train.rename(columns={"question": "topic"})
        df_test = df_test.rename(columns={"question": "topic"})

        topic = df_test["topic"].value_counts().index[0]
        t_topic = time.time()

        vec = GloveVectorizer()

        A1_train = vec.transform(df_train["a1"])
        A2_train = vec.transform(df_train["a2"])

        X_train = A2_train - A1_train

        y_train = df_train["y"]
        clf = svm.SVC(kernel="linear", C=0.1, probability=True)
        clf.fit(X_train, y_train)

        A1_test = vec.transform(df_test["a1"])
        A2_test = vec.transform(df_test["a2"])
        X_test = A2_test - A1_test

        y_test = df_test["y"]
        df_test["predicted_label"] = clf.predict(X_test)
        df_test["pred_score_1_soft"] = clf.predict_proba(X_test)[:, 1]
        # print("\t\t"+"; ".join([vec.get_feature_names()[i] for i in clf.coef_.toarray().argsort()[0][::-1][:5]]))
        # print("\t\t"+"; ".join([vec.get_feature_names()[i] for i in clf.coef_.toarray().argsort()[0][:5]]))

        print(
            "***\t time for fold {}: {:}; accuracy={}".format(
                topic,
                utils.format_time(time.time() - t_fold),
                np.round(
                    accuracy_score(
                        y_true=df_test["y"], y_pred=df_test["predicted_label"]
                    ),
                    2,
                ),
            )
        )

        df_test_all = pd.concat([df_test_all, df_test])

    fname = os.path.join(
        data_loaders.BASE_DIR,
        "tmp",
        "df_test_all_ArgGlove_cross_topic_validation_{}_{}.csv".format(
            CROSS_TOPIC, data_source
        ),
    )
    df_test_all.to_csv(fname)

    df_test_all["dataset"] = data_source
    df_results_all = pd.concat([df_results_all, df_test_all])
    print(
        "*** time for {}: {:}".format(
            data_source, utils.format_time(time.time() - t_source)
        )
    )

print("*** time for ArgGlove: {:}".format(utils.format_time(time.time() - total_t0)))
fname = os.path.join(
    data_loaders.BASE_DIR,
    "tmp",
    "df_results_all_ArgGlove_cross_topic_validation_{}.csv".format(CROSS_TOPIC),
)
df_results_all.to_csv(fname)
