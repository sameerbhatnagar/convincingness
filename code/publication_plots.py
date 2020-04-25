import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import data_loaders, weighted_scorers
from scipy.special import softmax

CROSS_TOPIC = True


def melt_results(means_df, id_var="dataset"):

    x = means_df.reset_index()[["acc", "AUC", id_var]].melt(id_vars=[id_var])
    y = means_df.reset_index()[["+/-(acc)", "+/-(AUC)", id_var]].melt(id_vars=[id_var])
    y["variable"] = y["variable"].map({"+/-(acc)": "acc", "+/-(AUC)": "AUC"})
    y = y.rename(columns={"value": "+/-"})
    means_df_melted = pd.merge(x, y, on=[id_var, "variable"])

    return means_df_melted


def argLength():
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

    # dalite
    train_dataframes, test_dataframes, _ = data["dalite"]
    df_test_all = pd.DataFrame()

    for i, (df_train, df_test) in enumerate(zip(train_dataframes, test_dataframes)):
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
        X_test = np.array(list(zip(df_test["a1_wc"].values, df_test["a2_wc"].values)))
        y_test = df_test["y"]
        df_test["predicted_label"] = clf.predict(X_test)
        df_test["pred_score_1_soft"] = clf.predict_proba(X_test)[:, 1]
        df_test["fold"] = i
        df_test_all = pd.concat([df_test_all, df_test])

    df_test_all_arglength = df_test_all
    df_test_all_arglength["dataset"] = "dalite"
    df_test_all["question"] = df_test_all["topic"]
    means_argLength_dalite = (
        df_test_all.groupby(["discipline", "question"])
        .apply(weighted_scorers.my_scores)
        .groupby("discipline")
        .apply(weighted_scorers.weighted_avg)
        .round(2)
    )

    means_argLength_dalite["dataset"] = "dalite"
    means_argLength_dalite = means_argLength_dalite.reset_index()

    means_argLength_dalite_plot = melt_results(
        means_argLength_dalite, id_var="discipline"
    )
    means_argLength_dalite_plot["model"] = "ArgLength"

    means_all = pd.DataFrame()
    for data_source in data.keys():
        train_dataframes, test_dataframes, _ = data[data_source]

        df_test_all = pd.DataFrame()

        for i, (df_train, df_test) in enumerate(zip(train_dataframes, test_dataframes)):
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

        df_test_all["dataset"] = data_source
        df_test_all_arglength = pd.concat([df_test_all_arglength, df_test_all])

        means_argLength = df_test_all.groupby(["topic"]).apply(
            weighted_scorers.my_scores
        )
        means_argLength["dataset"] = data_source
        means_all = pd.concat([means_all, means_argLength])

    means_argLength_other = (
        means_all.groupby("dataset").apply(weighted_scorers.weighted_avg).round(2)
    )
    means_argLength = pd.concat([means_argLength, means_argLength_other.reset_index()])
    means_argLength["model"] = "ArgLength"
    means_argLength_plot = melt_results(means_argLength_other)
    means_argLength_plot["model"] = "ArgLength"

    return means_argLength_dalite_plot, means_argLength_plot


def assemble_results():

    means_argLength_dalite_plot, means_argLength_plot = argLength()

    # ArgBoW
    print("ArgBow")
    fname = os.path.join(
        data_loaders.BASE_DIR,
        "tmp",
        "df_results_all_ArgBow_cross_topic_validation_{}.csv".format(CROSS_TOPIC),
    )
    df_results_all = pd.read_csv(fname)

    means = df_results_all.groupby(["dataset", "topic"]).apply(
        weighted_scorers.my_scores
    )
    means_df_bow = (
        means.groupby("dataset")
        .apply(weighted_scorers.weighted_avg)
        .round(2)
        .reset_index()
    )

    means_df_bow_melt = melt_results(means_df_bow)
    means_df_bow_melt["model"] = "ArgBoW"
    means_argLength_argBoW_plot = pd.concat([means_argLength_plot, means_df_bow_melt])

    df_results_all_dalite = df_results_all[df_results_all["dataset"] == "dalite"]
    means_dalite = df_results_all_dalite.groupby(["discipline", "topic"]).apply(
        weighted_scorers.my_scores
    )
    means_df_bow_dalite = (
        means_dalite.groupby("discipline")
        .apply(weighted_scorers.weighted_avg)
        .round(2)
        .reset_index()
    )

    means_df_plot_dalite_disc_bow = melt_results(
        means_df_bow_dalite, id_var="discipline"
    )
    means_df_plot_dalite_disc_bow["model"] = "ArgBoW"

    means_argLength_argBoW_plot_dalite = pd.concat(
        [means_argLength_dalite_plot, means_df_plot_dalite_disc_bow]
    )

    # ArgGlove
    print("ArgGlove")
    df_results_all = pd.DataFrame()
    for data_source in ["IBM_ArgQ", "IBM_Evi", "UKP", "dalite"]:
        fname = os.path.join(
            data_loaders.BASE_DIR,
            "tmp",
            "df_test_all_ArgGlove_cross_topic_validation_True_{}.csv".format(
                data_source
            ),
        )
        df_test_all_data_source = pd.read_csv(fname)
        df_test_all_data_source["dataset"] = data_source
        print("{}:{} rows".format(data_source, df_test_all_data_source.shape[0]))
        df_results_all = pd.concat([df_results_all, df_test_all_data_source])

    means_glove = df_results_all.groupby(["dataset", "topic"]).apply(
        weighted_scorers.my_scores
    )
    means_df_glove = (
        means_glove.groupby("dataset")
        .apply(weighted_scorers.weighted_avg)
        .round(2)
        .reset_index()
    )
    means_df_glove_melt = melt_results(means_df_glove)
    means_df_glove_melt["model"] = "ArgGlove"
    means_df_plot_glove = pd.concat([means_argLength_argBoW_plot, means_df_glove_melt])

    df_results_all_dalite = df_results_all[df_results_all["dataset"] == "dalite"]
    means_dalite_glove = df_results_all_dalite.groupby(["discipline", "topic"]).apply(
        weighted_scorers.my_scores
    )
    means_df_glove_dalite = (
        means_dalite_glove.groupby("discipline")
        .apply(weighted_scorers.weighted_avg)
        .round(2)
        .reset_index()
    )

    means_df_plot_dalite_disc_glove = melt_results(
        means_df_glove_dalite, id_var="discipline"
    )
    means_df_plot_dalite_disc_glove["model"] = "ArgGlove"
    means_df_plot_dalite_disc_glove = pd.concat(
        [means_argLength_argBoW_plot_dalite, means_df_plot_dalite_disc_glove]
    )

    # ArgBERT
    print("ArgBERT")
    df_results_bert_all = pd.DataFrame()

    for data_source in ["IBM_ArgQ", "IBM_Evi", "UKP"]:
        results_dir = os.path.join(
            data_loaders.BASE_DIR,
            "tmp",
            "results_cross_topic",
            "BERT_{}_results".format(data_source),
        )
        for fold in range(len(os.listdir(results_dir))):
            fname = os.path.join(
                results_dir, "fold-{}".format(fold), "df_test-fold-{}.csv".format(fold)
            )
            df_test = pd.read_csv(fname)
            df_test["dataset"] = data_source
            df_results_bert_all = pd.concat([df_results_bert_all, df_test])

    data_source = "dalite"
    for discipline in ["Biology", "Chemistry", "Physics"]:
        results_dir = os.path.join(
            data_loaders.BASE_DIR,
            "tmp",
            "results_cross_topic",
            "BERT_{}_results".format(data_source),
            discipline,
        )
        for fold in range(len(os.listdir(results_dir))):
            fname = os.path.join(
                results_dir, "fold-{}".format(fold), "df_test-fold-{}.csv".format(fold)
            )
            df_test = pd.read_csv(fname)
            df_test["dataset"] = data_source
            df_test["discipline"] = discipline
            df_results_bert_all = pd.concat([df_results_bert_all, df_test])

    df_results_bert_all["pred_score_1_soft"] = softmax(
        df_results_bert_all[["pred_score_0", "pred_score_1"]].values, axis=1
    )[:, 1]

    df_results_bert_all.groupby("dataset").size()

    means_bert = df_results_bert_all.groupby(["dataset", "topic"]).apply(
        weighted_scorers.my_scores
    )
    means_df_bert = (
        means_bert.groupby("dataset")
        .apply(weighted_scorers.weighted_avg)
        .round(2)
        .reset_index()
    )
    means_df_bert_melt = melt_results(means_df_bert)
    means_df_bert_melt["model"] = "ArgBERT"
    # means_df_bert
    means_df_plot_all = pd.concat([means_df_plot_glove, means_df_bert_melt])

    df_results_bert_all_dalite = df_results_bert_all[
        df_results_bert_all["dataset"] == "dalite"
    ]
    means_dalite_bert = df_results_bert_all_dalite.groupby(
        ["discipline", "topic"]
    ).apply(weighted_scorers.my_scores)
    means_df_bert_dalite = (
        means_dalite_bert.groupby("discipline")
        .apply(weighted_scorers.weighted_avg)
        .round(2)
        .reset_index()
    )

    means_df_plot_dalite_disc_bert = melt_results(
        means_df_bert_dalite, id_var="discipline"
    )
    means_df_plot_dalite_disc_bert["model"] = "ArgBERT"
    # means_df_plot_dalite_disc_bert
    means_df_plot_dalite_disc = pd.concat(
        [means_df_plot_dalite_disc_glove, means_df_plot_dalite_disc_bert]
    )

    return means_df_plot_all, means_df_plot_dalite_disc
