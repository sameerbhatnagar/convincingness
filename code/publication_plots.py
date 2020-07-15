import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import data_loaders, weighted_scorers
from scipy.special import softmax

CROSS_TOPIC = True

import matplotlib.pyplot as plt
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# https://stackoverflow.com/a/42033734
def grouped_barplot(df, cat,subcat, val ,ylabel, err,bar_colors,fname=None):
    plt.style.use("ggplot")
    u = df[cat].unique()
    x = np.arange(len(u))
    # print(x)
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    # print(offsets)
    width= np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(
            x+offsets[i],
            dfg[val].values,
            width=width,
            # label="{} {}".format(subcat, gr),
            label="{}".format(gr),
            yerr=dfg[err].values,
            color=bar_colors[i],
            error_kw=dict(ecolor='gray',capsize=3),
            )
    plt.xlabel(cat)
    plt.ylabel(ylabel)
    plt.ylim((0.5,1))
    plt.xticks(x, u)
    # plt.axhline(y=0.83,color="black",linewidth=8)
    plt.legend()
    if fname:
      plt.savefig(fname)
    else:
      plt.show()


def grouped_barplot_by_dataset(df, cat,subcat, val ,ylabel, err,bar_colors,fname=None):
    """
    e.g.
    means_df_plot_all2.groupby("variable").apply(
    lambda x: grouped_barplot_by_dataset(
        x,
        cat="dataset",
        subcat="model",
        val="value",
        ylabel=x["variable"].iat[0],
        bar_colors=["cornflowerblue","lightsteelblue","darkorange","mediumturquoise"],
        err="+/-",
        )
    )
    """

    hatch = ["","-",".","///"]
    plt.style.use("ggplot")
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()

    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(
            x+offsets[i],
            dfg[val].values,
            width=width,
            label="{}".format(gr),
            yerr=dfg[err].values,
            linewidth=dfg["sig"].values,
            edgecolor=['black']*len(x), #bug for linewidth: https://github.com/matplotlib/matplotlib/issues/9351
            color=bar_colors,
            hatch=hatch[i],
            error_kw=dict(ecolor='gray',capsize=3),
            )
    plt.xlabel(cat)
    plt.ylabel(ylabel)
    plt.ylim((0.5,1))
    plt.xticks(x, u)
    plt.legend()
    if fname:
      plt.savefig(fname)
    else:
      plt.show()

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
            data_source=data_source,
            cross_topic_validation=CROSS_TOPIC
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


def get_unique_args(df):
  return len(list(set(df["a1_id"].value_counts().index.to_list()+df["a2_id"].value_counts().index.to_list())))


def get_corpus(df):
  all_args_train = pd.concat(
          [
           df[["a1_id","a1"]].rename(columns={"a1_id":"id","a1":"a"}),
           df[["a2_id","a2"]].rename(columns={"a2_id":"id","a2":"a"})
          ]
          )

  corpus = all_args_train.drop_duplicates("id")["a"]
  return corpus

def get_vocab(corpus):
  vec=get_vectorizer()
  vec.fit(corpus)
  return(vec.get_feature_names())

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def get_vectorizer(term_freq=False,lemmatize=False,idf=True):
  if term_freq:
    if lemmatize:
      return TfidfVectorizer(tokenizer=LemmaTokenizer(),use_idf=idf,token_pattern=None)
    else:
      return TfidfVectorizer(use_idf=idf)
  else:
    if lemmatize:
      return CountVectorizer(tokenizer=LemmaTokenizer(),token_pattern=None)
    else:
      return CountVectorizer()


def make_data_summary_table(fpath):
    """
    produce table which give N_args,N_pairs, and other overall stats for dataset
    """
    CROSS_TOPIC=False
    N_FOLDS=10
    data = {}
    data_sources=["IBM_ArgQ","UKP"]
    for data_source in data_sources:
      data[data_source] = data_loaders.load_arg_pairs(
          data_source=data_source,
          N_folds=N_FOLDS,
          cross_topic_validation=CROSS_TOPIC,
          train_test_split=False
      )

    for discipline in ["Physics","Biology","Chemistry","Ethics"]:
        data[discipline] = data_loaders.load_dalite_data(
            discipline=discipline,
            N_folds=N_FOLDS,
            cross_topic_validation=CROSS_TOPIC,
            train_test_split=False
        )
    data["IBM_Evi"] = data_loaders.load_arg_pairs_IBM_Evi(
        N_folds=N_FOLDS,
        cross_topic_validation=CROSS_TOPIC,
        train_test_split=False
    )
    summary=[]
    for data_source in ["IBM_ArgQ","UKP","IBM_Evi","Physics","Biology","Chemistry","Ethics"]:
        d={}
    #     _,_,df_all=data[data_source]
        df_all=data[data_source]
        df_all = df_all.rename(columns={"question":"topic"})
        d["dataset"]=data_source
        d["N_pairs"]=df_all.shape[0]
        try:
            d["N_topics"]=df_all["topic"].value_counts().shape[0]
        except AttributeError:
            d["N_topics"]=df_all["topic"].iloc[:,0].value_counts().shape[0]
    #     if data_source!="dalite":
    #     df_all["a1_id"]=df_all["#id"].str.split("_").apply(lambda x: x[0])
    #     df_all["a2_id"]=df_all["#id"].str.split("_").apply(lambda x: x[1])

        d["N_args"]=get_unique_args(df_all)

        corpus = get_corpus(df_all)
        d["Vocab"] = len(get_vocab(corpus))

        # d["Args/topic"],d["Args/topic_std"]=df_all.groupby("topic").apply(lambda x: get_unique_args(x)).describe()[["mean","std"]].astype(int)

        m,s=pd.concat(
          [
                 df_all[["a1_id","a1"]].rename(columns={"a1_id":"id","a1":"a"}),
                 df_all[["a2_id","a2"]].rename(columns={"a2_id":"id","a2":"a"})
          ]
        ).drop_duplicates("id")["a"].str.count("\w+").describe()[["mean","std"]].astype(int)

        d["wc_mean (SD)"]= "{} ({})".format(m,s)

        m,s=np.abs(
              df_all["a1"].str.count("\w+")-df_all["a2"].str.count("\w+")
              ).describe()[["mean","std"]].astype(int)

        d["wc_diff_mean (SD)"]="{} ({})".format(m,s)

        summary.append(d)

    df_summary=pd.DataFrame(summary).set_index("dataset")

    print("saving df_summary to {}".format(fpath))
    df_summary.to_latex(fpath)

    return df_summary
