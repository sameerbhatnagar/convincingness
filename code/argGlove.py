import data_loaders
import utils
import pandas as pd
import time
import numpy as np
import os
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm
from sklearn.metrics import accuracy_score

import plac

import spacy

NLP_MD = spacy.load("en_core_web_md", disable=["ner", "tagger", "parser"])

class GloveVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._nlp = NLP_MD

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.concatenate([doc.vector.reshape(1, -1) for doc in self._nlp.pipe(X)])

@plac.annotations(
    N_folds=plac.Annotation("N_folds",kind="option"),
    cross_topic_validation=plac.Annotation("cross_topic_validation",kind="flag"),
    data_source=plac.Annotation("data_source"),
)

def main(cross_topic_validation,data_source,N_folds=5):

    if data_source == "all":
        data_sources = ["IBM_ArgQ","UKP","IBM_Evi","dalite"]
    else:
        data_sources = [data_source]


    data = {}
    for data_source in data_sources:
        data[data_source] = data_loaders.load_arg_pairs(
            data_source=data_source,
            N_folds=N_folds,
            cross_topic_validation=cross_topic_validation,
        )

    total_t0 = time.time()

    # save results to:
    if cross_topic_validation:
        results_sub_dir="cross_topic_validation"
    else:
        results_sub_dir="{}_fold_validation".format(N_folds)

    fname="df_results_all_ArgGlove_{}.csv".format("_".join(data_sources))

    fpath = os.path.join(
        data_loaders.BASE_DIR,
        os.pardir,
        "tmp",
        results_sub_dir,
        fname
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Saving results to {}".format(fname))


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

            if cross_topic_validation:
                fold_name = df_test["topic"].value_counts().index[0]
            else:
                fold_name=i
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
                    fold_name,
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

        # fname = os.path.join(
        #     data_loaders.BASE_DIR,
        #     "tmp",
        #     "df_test_all_ArgGlove_cross_topic_validation_{}_{}.csv".format(
        #         CROSS_TOPIC, data_source
        #     ),
        # )
        # df_test_all.to_csv(fname)

        df_test_all["dataset"] = data_source
        df_results_all = pd.concat([df_results_all, df_test_all])
        print(
            "*** time for {}: {:}".format(
                data_source, utils.format_time(time.time() - t_source)
            )
        )

    print("*** time for ArgGlove: {:}".format(utils.format_time(time.time() - total_t0)))

    df_results_all.to_csv(fname)


if __name__ == '__main__':
    import plac; plac.call(main)
