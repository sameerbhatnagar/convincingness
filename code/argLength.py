import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import data_loaders, weighted_scorers,utils
import time
import plac
from pathlib import Path


@plac.annotations(
    N_folds=plac.Annotation("N_folds",kind="option"),
    cross_topic_validation=plac.Annotation("cross_topic_validation",kind="flag"),
)


def main(cross_topic_validation,N_folds=5):

    data_sources = ["IBM_ArgQ","UKP","IBM_Evi","dalite"]

    data = {}
    for data_source in data_sources:
        data[data_source] = data_loaders.load_arg_pairs(
            data_source=data_source,
            N_folds=N_folds,
            cross_topic_validation=cross_topic_validation,
        )

    for discipline in ["Physics","Chemistry","Biology"]:
        data[discipline] = data_loaders.load_arg_pairs(
            data_source="dalite",
            N_folds=N_folds,
            cross_topic_validation=cross_topic_validation,
            discipline=discipline,
        )

    total_t0 = time.time()

    model_name="ArgLength"

    # save results to:
    if cross_topic_validation:
        results_sub_dir=os.path.join(
            data_loaders.BASE_DIR,
            "tmp",
            model_name,
            "cross_topic_validation"
            )
    else:
        results_sub_dir=os.path.join(
            data_loaders.BASE_DIR,
            "tmp",
            model_name,
            "{}_fold_validation".format(N_folds)
            )

    Path(results_sub_dir).mkdir(parents=True, exist_ok=True)

    df_test_all_arglength = pd.DataFrame()
    for data_source in data.keys():
        print(data_source)
        train_dataframes, test_dataframes, _ = data[data_source]

        df_test_all = pd.DataFrame()

        for i, (df_train, df_test) in enumerate(zip(train_dataframes, test_dataframes)):
            print("Fold {}".format(i),end=",")
            df_train = df_train.rename(columns={"question": "topic"})
            df_test = df_test.rename(columns={"question": "topic"})
            df_train["a1_wc"] = df_train["a1"].str.count("\w+")
            df_train["a2_wc"] = df_train["a2"].str.count("\w+")
            df_test["a1_wc"] = df_test["a1"].str.count("\w+")
            df_test["a2_wc"] = df_test["a2"].str.count("\w+")
            df_train["y"] = df_train["label"].map({"a1": -1, "a2": 1})
            df_test["y"] = df_test["label"].map({"a1": -1, "a2": 1})

            # sns.scatterplot(x="a1_wc",y="a2_wc",hue="label",data=df_train)
            X_train = np.array(
                list(zip(df_train["a1_wc"].values, df_train["a2_wc"].values))
            )
            y_train = df_train["y"]
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            X_test = np.array(
                list(zip(df_test["a1_wc"].values, df_test["a2_wc"].values))
            )
            y_test = df_test["y"]
            df_test["predicted_label"] = clf.predict(X_test)
            df_test["pred_score_1_soft"] = clf.predict_proba(X_test)[:, 1]
            df_test["fold"] = i
            df_test_all = pd.concat([df_test_all, df_test])

        # df_test_all["dataset"] = data_source
        # df_test_all_arglength = pd.concat([df_test_all_arglength, df_test_all])

        fname="df_results_all_ArgLength_{}.csv".format(data_source)

        fpath = os.path.join(
            results_sub_dir,
            fname
        )
        print("\n Saving results to {}".format(fpath))

        df_test_all.to_csv(fpath)

    print("*** time for ArgLength: {:}".format(utils.format_time(time.time() - total_t0)))

if __name__ == '__main__':
    import plac; plac.call(main)
