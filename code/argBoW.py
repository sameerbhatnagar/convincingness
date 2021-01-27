import os
from pathlib import Path
import plac
import pandas as pd
import time

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm

import data_loaders, utils




@plac.annotations(
    N_folds=plac.Annotation("N_folds", kind="option"),
    cross_topic_validation=plac.Annotation("cross_topic_validation", kind="flag"),
    data_source=plac.Annotation("data_source"),
)

def main(cross_topic_validation, data_source, N_folds=5):
    data={}
    if data_source in ["Physics","Chemistry","Biology","Ethics"]:
        data[data_source] = data_loaders.load_dalite_data(
            N_folds=N_folds,
            cross_topic_validation=cross_topic_validation,
            discipline=data_source,
        )
    else:
        data[data_source] = data_loaders.load_arg_pairs(
            data_source=data_source,
            N_folds=N_folds,
            cross_topic_validation=cross_topic_validation,
        )

    lemmatize = True
    term_freq = True
    idf = False

    total_t0 = time.time()

    model_name = "ArgBoW"

    # save results to:
    if cross_topic_validation:
        results_sub_dir = os.path.join(
            data_loaders.BASE_DIR, "tmp", model_name, "cross_topic_others_validation"
        )
    else:
        results_sub_dir = os.path.join(
            data_loaders.BASE_DIR,
            "tmp",
            model_name,
            "{}_fold_validation".format(N_folds),
        )

    Path(results_sub_dir).mkdir(parents=True, exist_ok=True)

    df_results_all = pd.DataFrame()
    meta = []

    for data_source in data.keys():
        print(data_source)
        t_source = time.time()

        train_dataframes, test_dataframes, df_all = data[data_source]
        df_all = df_all.rename(columns={"question": "topic"})

        for i, (df_train, df_test) in enumerate(zip(train_dataframes, test_dataframes)):

            df_train["y"] = df_train["label"].map({"a1": -1, "a2": 1})
            df_test["y"] = df_test["label"].map({"a1": -1, "a2": 1})
            # df_train = df_train.rename(columns={"question": "topic"})
            # df_test = df_test.rename(columns={"question": "topic"})
            topic = df_test["topic"].value_counts().index[0]
            print("Fold {}".format(i), end=",")
            t_topic = time.time()

            corpus = utils.get_corpus(df=df_train)

            vec = utils.get_vectorizer(term_freq=term_freq, lemmatize=lemmatize, idf=idf)
            vec.fit(corpus)
            # vocab=vec.get_feature_names()
            # d["vocab_train"]=len(vocab)

            A1_train = vec.transform(df_train["a1"])
            A2_train = vec.transform(df_train["a2"])

            X_train = A2_train - A1_train
            # d["train_obs"]=X_train.shape[0]
            # d["train_obs_feat"]=X_train.shape[1]

            y_train = df_train["y"]
            clf = svm.SVC(kernel="linear", C=0.1, probability=True)
            clf.fit(X_train, y_train)

            # vocab_test = get_vocab(get_corpus(df=df_test))
            # d["vocab_test"]=len(vocab_test)

            # unknown_tokens = [w for w in vocab_test if w not in vocab]
            # unknown_tokens=[t.replace(")","").replace("(","") for t in unknown_tokens]
            # d["unknown"]=len(unknown_tokens)

            # pattern ="|".join(unknown_tokens)

            A1_test = vec.transform(df_test["a1"])  # .str.replace(pattern,"<UNK>"))
            A2_test = vec.transform(df_test["a2"])  # .str.replace(pattern,"<UNK>"))
            X_test = A2_test - A1_test
            # d["test_obs"] = X_test.shape[0]
            # d["test_obs_feat"] = X_test.shape[1]

            y_test = df_test["y"]
            df_test["predicted_label"] = clf.predict(X_test)
            df_test["pred_score_1_soft"] = clf.predict_proba(X_test)[:, 1]
            # print("\t\t"+"; ".join([vec.get_feature_names()[i] for i in clf.coef_.toarray().argsort()[0][::-1][:5]]))
            # print("\t\t"+"; ".join([vec.get_feature_names()[i] for i in clf.coef_.toarray().argsort()[0][:5]]))
            # meta.append(d)
            df_test["dataset"] = data_source

            df_results_all = pd.concat([df_results_all, df_test])

            print(
                "*** time for {}: {:}".format(
                    topic, utils.format_time(time.time() - t_topic)
                )
            )

        fname = "df_results_all_{}_{}.csv".format(model_name, data_source)

        fpath = os.path.join(results_sub_dir, fname)
        print("\n Saving results to {}".format(fpath))

        df_results_all.to_csv(fpath)

    print(
        "*** time for {}: {:}".format(data_source, utils.format_time(time.time() - t_source))
    )


if __name__ == '__main__':
    import plac; plac.call(main)
