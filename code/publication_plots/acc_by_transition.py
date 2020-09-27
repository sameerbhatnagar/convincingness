import os, json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import iqr

import spacy

nlp = spacy.load("en_core_web_sm")
DROPPED_POS = ["PUNCT", "SPACE"]

import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# careful if dir names change
RESULTS_DIR = os.path.join(BASE_DIR, "tmp", "measure_convincingness")

def get_topic_data(topic, discipline):
    """
    given topic/question and associated discipline (needed for subdirectories),
    return mydalite answer observations, and associated pairs that are
    constructed using `mauke_pairs.py`
    """

    data_dir_discipline = get_data_dir(discipline)

    fp = os.path.join(data_dir_discipline, "{}.csv".format(topic))
    df_topic = pd.read_csv(fp)
    df_topic = df_topic[~df_topic["user_token"].isna()].sort_values("a_rank_by_time")
    df_topic["rationale"] = df_topic["rationale"].fillna(" ")
    # load pairs
    pairs_df = pd.read_csv(
        os.path.join(
            "{}_pairs".format(data_dir_discipline), "pairs_{}.csv".format(topic)
        )
    )
    pairs_df = pairs_df[pairs_df["a1_id"] != pairs_df["a2_id"]]

    return pairs_df, df_topic

def get_data_dir(discipline):
    return os.path.join(RESULTS_DIR, discipline, "data")

def my_summary(x):
    d = {}
    d["N"] = "{:0.0f}".format(x["n"].sum())
    d["acc"] = np.average(x["acc"], weights=x["n"])
    d["std"] = np.std(x["acc"])

    return pd.Series(d, index=["N", "acc", "std"])


def my_summary_table(x):
    d = {}
    d["N"] = x["N"].sum().astype(int)
    d["N_pairs"] = x["N_pairs"].sum().astype(int)
    #     d["wc_mean"]="{} ({})".format(
    #         np.average(x["mean_wc"],weights=x["N"]).round(0).astype(int),
    #         np.average(x["std_wc"],weights=x["N"]).round(0).astype(int),
    #     )
    d["wc_median"] = "{} ({})".format(
        x["median_wc"].median().round(0).astype(int),
        iqr(x["iqr_wc"]).round(0).astype(int),
    )

    return pd.Series(d, index=["N", "N_pairs", "wc_median"])


def summary_table():
    """
    function to give descriptive statistics of dataset, and output table to
    latex file in article sub-folder
    """
    discipline="Physics"
    transitions = ["rr","wr","rw","ww"]
    d_summary = []
    topics = os.listdir(os.path.join(RESULTS_DIR, discipline, "data"))
    for i, topic in enumerate(topics):
        if i % 25 == 0:
            print("{}/{} topics done".format(i, len(topics)))
        pairs_df, df_topic = get_topic_data(
            topic=topic.replace(".csv", ""), discipline=discipline
        )
        counts = df_topic["transition"].value_counts().to_dict()
        counts_pairs = pairs_df["transition"].value_counts().to_dict()

        df_wc = pd.DataFrame(
            [
                (
                    transition,
                    len([token for token in doc if token.pos_ not in DROPPED_POS]),
                )
                for doc, transition in nlp.pipe(
                    zip(
                        df_topic["rationale"].to_list(),
                        df_topic["transition"].to_list(),
                    ),
                    as_tuples=True,
                    batch_size=20,
                )
            ]
        ).rename(columns={0: "transition", 1: "wc"})

        means_wc = df_wc.groupby("transition")["wc"].mean().to_dict()
        stds_wc = df_wc.groupby("transition")["wc"].std().to_dict()
        medians_wc = df_wc.groupby("transition")["wc"].median().to_dict()
        iqr_wc = df_wc.groupby("transition")["wc"].apply(lambda x: iqr(x)).to_dict()

        for transition in transitions:
            d = {}
            d["topic"] = topic
            d["transition"] = transition
            d["N"] = counts.get(transition, 0)
            d["N_pairs"] = counts_pairs.get(transition, 0)
            d["mean_wc"] = means_wc.get(transition)
            d["std_wc"] = stds_wc.get(transition)
            d["median_wc"] = medians_wc.get(transition)
            d["iqr_wc"] = iqr_wc.get(transition)
            d_summary.append(d)

    df_summary = pd.DataFrame(d_summary)
    df_summary_table = (
        df_summary.dropna().groupby("transition").apply(lambda x: my_summary_table(x))
    )

    fp = os.path.join(BASE_DIR, "articles", "lak2021", "data", "df_summary.tex")
    df_summary_table.to_latex(fp)
    print(fp)
    return


def main():

    summary_table()

    ### Load data
    print("loading data")
    discipline = "Physics"
    df = pd.DataFrame()
    rank_scores, acc_trans = {}, {}
    rank_score_types = ["wc", "winrate", "elo", "crowd_BT", "BT"]
    for i,rank_score_type in enumerate(rank_score_types,1):
        print(rank_score_type)
        results_dir_discipline = os.path.join(
            RESULTS_DIR,
            discipline,
            "model_fit",
            rank_score_type,
        )
        topics = os.listdir(
            os.path.join(
                results_dir_discipline,
                "accuracies"
            )
        )
        df_acc=pd.DataFrame()
        acc_trans[rank_score_type]={}
        rank_scores[rank_score_type]={}
        for j,topic in enumerate(topics,1):

            fp=os.path.join(
                results_dir_discipline,
                "accuracies",
                "{}".format(topic)
            )
            with open(fp,"r") as f:
                d=json.load(f)
            d2={k:v for k,v in d.items() if k!="acc_by_transition"}
            d2.update({"transition":"all","topic":topic})
            df_acc_t=pd.DataFrame(d2,index=[i*j])

            df_acc=pd.concat([df_acc,df_acc_t])

            acc_trans[rank_score_type][topic]={k:v for k,v in d.items() if k=="acc_by_transition"}

            fp=os.path.join(
                results_dir_discipline,
                "rank_scores",
                "{}".format(topic)
            )
            with open(fp,"r") as f:
                rank_scores[rank_score_type][topic.replace(".json","")]=json.load(f)

        df_acc=df_acc.dropna()
        df_acc["rank_score_type"]=rank_score_type
        df=pd.concat([df,df_acc])


    df.loc[df["rank_score_type"]=="baseline","rank_score_type"]="WinRate"
    df.loc[df["rank_score_type"]=="wc","rank_score_type"]="WordCount"

    # collect rank_scores
    df_all=pd.DataFrame()
    for rank_score_type in rank_score_types:
        df_all_topics=pd.DataFrame()
        for topic,scores in rank_scores[rank_score_type].items():
            df_scores=pd.DataFrame.from_dict(
                rank_scores[rank_score_type][topic],
                orient="index"
            ).reset_index().rename(
                columns={
                    0:"value",
                    "index":"arg_id"
                }
            )
            df_scores["topic"]=topic
            df_all_topics=pd.concat([df_all_topics,df_scores])
        df_all_topics["rank_score_type"]=rank_score_type
        df_all=pd.concat([df_all,df_all_topics])

    # collect different scores for each arg
    print("collecting different scores for each arg")
    df_pivot = pd.pivot(df_all, index="arg_id", columns="rank_score_type")["value"]
    topics = pd.pivot(df_all, index="arg_id", columns="rank_score_type")["topic"]
    df_pivot = pd.merge(
        df_pivot,
        topics[["wc"]].rename(columns={"wc": "topic"}),
        left_index=True,
        right_index=True,
    )

    # append transition type for each arg
    print("loading transition data for each arg")
    data_dir = os.path.join(RESULTS_DIR, discipline, "data")
    df_topics = pd.DataFrame()
    for topic, df_pivot_topic in df_pivot.groupby("topic"):
        fp = os.path.join(data_dir, "{}.csv".format(topic))
        df_topic = pd.read_csv(fp)
        df_topic["arg_id"] = "arg" + df_topic["id"].astype(str)
        df_topic = df_topic.set_index("arg_id")[["transition"]]
        df_topics = pd.concat([df_topics, df_topic])

    df_pivot["transition"] = df_pivot.index.map(df_topics["transition"].to_dict())

    # correlation plot for each transition type
    transitions = ["rr", "rw", "wr", "ww"]
    transition_labels = {
        "rr": "Right -> Right",
        "rw": "Right -> Wrong",
        "wr": "Wrong -> Right",
        "ww": "Wrong -> Wrong",
    }
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    for ax, transition in zip(axs.flatten(), transitions):
        corr = df_pivot.groupby("transition").corr().loc[transition]
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr.iloc[1:, :-1],
            mask=mask[1:, :-1],
            cmap=cmap,
            vmax=1.0,
            vmin=0.0,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )
        ax.text(2.25, 0.75, transition_labels[transition], size=14)
    for ax in axs.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.tight_layout()
    fp = os.path.join(BASE_DIR, "articles", "lak2021", "img", "corr_plot.pgf")
    print(fp)
    fig.savefig(fp)
    ### very very slow. load saved version into p as shortcut
    p = pd.DataFrame()
    for rank_score_type in rank_score_types:
        print(rank_score_type)
        for i, (topic, d_array) in enumerate(acc_trans[rank_score_type].items()):
            q=pd.DataFrame(d_array["acc_by_transition"])
            q["rank_score_type"]=rank_score_type
            q["topic"]=topic
            p=pd.concat([p,q])

    df2 = pd.concat([df, p.rename(columns={"n_shape": "n"})])

    #### accuracies for each rank score type by transition
    df_plot = (
        df2.dropna(subset=["acc"])
        .groupby(["rank_score_type", "transition"])
        .apply(lambda x: my_summary(x))
        .reset_index()
    )
    df_plot.loc[df_plot["rank_score_type"] == "WordCount", "rank_score_type"] = "wc"

    col_ordering = ["crowd_BT", "BT", "elo", "winrate", "wc"]
    transition_colors = {
        "all": "gray",
        "rr": "green",
        "rw": "yellow",
        "wr": "blue",
        "ww": "red",
    }
    fig, ax = plt.subplots()
    sns.set()
    for transition, df_transition in df_plot.groupby("transition"):
        df_t = df_transition.set_index("rank_score_type").reindex(col_ordering)
        plt.plot(
            df_t.index,
            df_t["acc"],
            marker="o",
            label=transition,
            alpha=0.5,
            color=transition_colors[transition],
        )
        plt.errorbar(
            x=df_t.index, y=df_t["acc"], yerr=df_t["std"], alpha=0.5,
        )
    plt.legend()
    fp = os.path.join(BASE_DIR, "articles", "lak2021", "img", "acc_by_transition.pgf")
    print(fp)
    fig.savefig(fp)


if __name__ == "__main__":
    import plac

    plac.call(main)
