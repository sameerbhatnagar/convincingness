import os, json, math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import iqr, kendalltau
import data_loaders

import spacy

nlp = spacy.load("en_core_web_sm")
DROPPED_POS = ["PUNCT", "SPACE"]
TRANSITIONS = {
    "Physics": ["rr", "rw", "wr", "ww"],
    "Chemistry": ["rr", "rw", "wr", "ww"],
    "Ethics": ["switch_ans", "same_ans"],
}
TRANSITION_LABELS = {
    "Physics": {
        "rr": "Right -> Right",
        "rw": "Right -> Wrong",
        "wr": "Wrong -> Right",
        "ww": "Wrong -> Wrong",
    },
    "Chemistry": {
        "rr": "Right -> Right",
        "rw": "Right -> Wrong",
        "wr": "Wrong -> Right",
        "ww": "Wrong -> Wrong",
    },
    "Ethics": {"switch_ans": "Switch", "same_ans": "Same"},
}
RANK_SCORE_TYPES = [
    "crowd_BT",
    # "crowdBT_filtered",
    "BT",
    "elo",
    "winrate",
    "winrate_no_pairs",
    "wc",
]
RANK_SCORE_TYPES_RENAMED = {
    "crowd_BT": "CrowdBT",
    # "crowdBT_filtered": "CrowdBT_f",
    "wc": "Length",
    "elo": "Elo",
    "winrate": "WinRate_pairs",
    "winrate_no_pairs": "WinRate",
    "BT": "BT",
}
RANK_SCORE_TYPE_COLORS = {
    "BT": "darkblue",
    "crowd_BT": "steelblue",
    "elo": "orange",
    "winrate": "purple",
    "winrate_no_pairs": "orchid",
    # "crowdBT_filtered": "cornflowerblue",
}
TRANSITION_COLORS = {
    "Physics": {
        "rr": "forestgreen",
        "rw": "gold",
        "wr": "cornflowerblue",
        "ww": "firebrick",
    },
    "Chemistry": {
        "rr": "forestgreen",
        "rw": "gold",
        "wr": "cornflowerblue",
        "ww": "firebrick",
    },
    "Ethics": {"switch_ans": "gold", "same_ans": "cornflowerblue",},
}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THESIS_DIR=os.path.join(BASE_DIR, os.pardir,"thesis_project", "thesis")


def get_topic_data(topic, discipline,output_dir):
    """
    given topic/question and associated discipline (needed for subdirectories),
    return mydalite answer observations, and associated pairs that are
    constructed using `mauke_pairs.py`
    """
    data_dir_discipline = os.path.join(output_dir,"data")

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


def my_summary(x):
    d = {}
    d["N"] = "{:0.0f}".format(x["n"].sum())
    d["acc"] = np.average(x["acc"], weights=x["n"])
    d["std"] = np.std(x["acc"])

    return pd.Series(d, index=["N", "acc", "std"])


def my_summary_table(x):
    """
    function to aggregate model-fit accuracy data by transition
    """
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


def summary_batches_by_transition(x):
    d = {}
    d["N"] = x["N"].sum().astype(int)
    d["acc"] = np.average(x["acc"], weights=x["N"]).round(2)
    d["std"] = np.std(x["acc"])
    return pd.Series(d, index=["N", "acc", "std"])


def summary_batches_by_transition_corr(x):
    d = {}
    d["N"] = x["N"].sum().astype(int)
    d["N_common"] = x["N_common"].sum().astype(int)
    d["r"] = np.average(x["r"], weights=x["N_common"]).round(2)
    d["std"] = np.std(x["r"]).round(2)
    return pd.Series(d, index=["N", "N_common", "r", "std"])


def summary_table(discipline,output_dir):
    """
    function to give descriptive statistics of dataset, and output table to
    latex file in article sub-folder
    """

    d_summary = []
    topics = os.listdir(os.path.join(output_dir, "data"))
    for i, topic in enumerate(topics):
        if i % 25 == 0:
            print("{}/{} topics done".format(i, len(topics)))
        pairs_df, df_topic = get_topic_data(
            topic=topic.replace(".csv", ""), discipline=discipline,
            output_dir=output_dir
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

        for transition in TRANSITIONS[discipline]:
            d = {}
            d["topic"] = topic
            d["transition"] = TRANSITION_LABELS[discipline][transition]
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

    return df_summary_table


def load_data_for_plots(discipline,output_dir):
    """
    load data for corr_plot and accuracy plot by transition.

    Returns:
    ========
        df: data frame with model_fit accuracies for all data
        acc_trans: dict with model_fit accuracies for each transition
        rank_scores: dict with rank scores of all types for each arg
    """

    df = pd.DataFrame()
    rank_scores, acc_trans = {}, {}

    for i, rank_score_type in enumerate(RANK_SCORE_TYPES, 1):
        print(rank_score_type)
        results_dir_discipline = os.path.join(
            output_dir, "model_fit", rank_score_type,
        )
        topics = os.listdir(os.path.join(results_dir_discipline, "accuracies"))
        df_acc = pd.DataFrame()
        acc_trans[rank_score_type] = {}
        rank_scores[rank_score_type] = {}
        for j, topic in enumerate(topics, 1):

            fp = os.path.join(results_dir_discipline, "accuracies", "{}".format(topic))
            with open(fp, "r") as f:
                d = json.load(f)
            d2 = {k: v for k, v in d.items() if k != "acc_by_transition"}
            d2.update({"transition": "all", "topic": topic})
            df_acc_t = pd.DataFrame(d2, index=[i * j])

            df_acc = pd.concat([df_acc, df_acc_t])

            acc_trans[rank_score_type][topic] = {
                k: v for k, v in d.items() if k == "acc_by_transition"
            }

            fp = os.path.join(results_dir_discipline, "rank_scores", "{}".format(topic))
            with open(fp, "r") as f:
                rank_scores[rank_score_type][topic.replace(".json", "")] = json.load(f)

        df_acc = df_acc.dropna()
        df_acc["rank_score_type"] = rank_score_type
        df = pd.concat([df, df_acc])

    return df, acc_trans, rank_scores


def draw_corr_plot(rank_scores, discipline, output_dir,transition=None):
    """
    draw correlation plot between different rank_score_types for each arg
    disaggregated by transition type

    Arguments:
    ==========
        rank_scores: dict of rank scores for each transition
        transition: optional, string, to get only one of four corr plots
    Returns
    =======
        fig: matplotlib figure object
    """

    # df.loc[df["rank_score_type"] == "baseline", "rank_score_type"] = "WinRate"
    # df.loc[df["rank_score_type"] == "wc", "rank_score_type"] = "WordCount"

    # collect rank_scores
    df_all = pd.DataFrame()
    for rank_score_type in RANK_SCORE_TYPES:
        df_all_topics = pd.DataFrame()
        for topic, scores in rank_scores[rank_score_type].items():
            df_scores = (
                pd.DataFrame.from_dict(
                    rank_scores[rank_score_type][topic], orient="index"
                )
                .reset_index()
                .rename(columns={0: "value", "index": "arg_id"})
            )
            df_scores["topic"] = topic
            df_all_topics = pd.concat([df_all_topics, df_scores])
        df_all_topics["rank_score_type"] = rank_score_type
        df_all = pd.concat([df_all, df_all_topics])

    # collect different scores for each arg
    # print("collecting different scores for each arg")
    df_pivot = pd.pivot(df_all, index="arg_id", columns="rank_score_type")["value"]
    topics = pd.pivot(df_all, index="arg_id", columns="rank_score_type")["topic"]
    df_pivot = pd.merge(
        df_pivot,
        topics[["wc"]].rename(columns={"wc": "topic"}),
        left_index=True,
        right_index=True,
    )

    # append transition type for each arg
    # print("loading transition data for each arg")
    data_dir = os.path.join(output_dir, "data")
    df_topics = pd.DataFrame()
    for topic, df_pivot_topic in df_pivot.groupby("topic"):
        fp = os.path.join(data_dir, "{}.csv".format(topic))
        df_topic = pd.read_csv(fp)
        df_topic["arg_id"] = "arg" + df_topic["id"].astype(str)
        df_topic = df_topic.set_index("arg_id")[["transition"]]
        df_topics = pd.concat([df_topics, df_topic])

    df_pivot["transition"] = df_pivot.index.map(df_topics["transition"].to_dict())
    ratios = (df_pivot["transition"].value_counts(normalize=True) * 100).astype(int)
    # correlation plot for each transition type
    if discipline!="Ethics":
        transition_labels = {
            "rr": "Right -> Right\n{}% of data".format(ratios["rr"]),
            "rw": "Right -> Wrong\n{}% of data".format(ratios["rw"]),
            "wr": "Wrong -> Right\n{}% of data".format(ratios["wr"]),
            "ww": "Wrong -> Wrong\n{}% of data".format(ratios["ww"]),
        }
    else:
        transition_labels = {
            "switch_ans":"Switch Answer Choice\n{}% of data".format(ratios["switch_ans"]),
            "same_ans":"Same Answer Choice\n{}% of data".format(ratios["same_ans"]),
        }
    df_pivot = df_pivot.rename(columns=RANK_SCORE_TYPES_RENAMED)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    if transition:
        fig, ax = plt.subplots()
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
        ax.set_yticklabels(corr.index[1:], va="center")
        ax.set_xlabel("")
        ax.set_ylabel("")

    else:
        figsize = (11, 9)
        if discipline!="Ethics":
            fig, axs = plt.subplots(2, 2, figsize=figsize)
        else:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        for ax, transition in zip(axs.flatten(), TRANSITIONS[discipline]):
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
    return fig


def draw_acc_by_transition(output_dir_name):
    """
    for each transition type, give accuracy of rank scores in
    pairwise prediction task
    """

    disciplines=["Physics","Chemistry","Ethics"]
    populations=["all","switchers"]

    fig,axs = plt.subplots(len(disciplines),len(populations),figsize=(12,9))

    for d,discipline in enumerate(disciplines):
        for p,population in enumerate(populations):

            ax=axs[d,p]
            output_dir=os.path.join(
                output_dir_name,
                discipline,
                population
            )

            df_results = pd.DataFrame()
            for rank_score_type in RANK_SCORE_TYPES:
                results_dir_discipline = os.path.join(
                    output_dir, "model_fit", rank_score_type, "accuracies"
                )
                # print(results_dir_discipline)
                topics = os.listdir(results_dir_discipline)
                for topic in topics:
                    fp = os.path.join(results_dir_discipline, topic)
                    with open(fp, "r") as f:
                        acc = json.load(f)

                    df_acc = pd.DataFrame(acc["acc_by_transition"])
                    df_acc["topic"]=topic
                    df_acc["rank_score_type"]=rank_score_type
                    df_results=pd.concat([df_results,df_acc])
            df_results=df_results.rename(columns={"n":"N"})

            df_table = (
                df_results.dropna().groupby(["rank_score_type", "transition"])
                .apply(lambda x: summary_batches_by_transition(x))
                .reset_index()
            )
            df_table["rank_score_type"] = pd.Categorical(
                df_table["rank_score_type"].map(RANK_SCORE_TYPES_RENAMED),
                [RANK_SCORE_TYPES_RENAMED[r] for r in RANK_SCORE_TYPES],
            )
            df_table = df_table.sort_values("rank_score_type")


            cat = "rank_score_type"
            subcat = "transition"
            val = "acc"
            err = "std"
            ylabel = "Pairwise Classification Accuracy"
            xlabel = "Rank Score Type"

            plt.style.use("ggplot")
            u = df_table[cat].unique()
            x = np.arange(len(u))
            subx = df_table[subcat].unique()

            tightness = 4.0
            offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (
                len(subx) + tightness
            )
            width = np.diff(offsets).mean()

    #         fig, ax = plt.subplots(figsize=(6, 4))
            for i, gr in enumerate(subx):
                dfg = df_table[df_table[subcat] == gr]
                ax.bar(
                    x + offsets[i],
                    dfg[val].values,
                    width=width,
                    label=TRANSITION_LABELS[discipline][gr],
                    yerr=dfg[err].values / 2,
                    color=TRANSITION_COLORS[discipline][gr],
                    error_kw=dict(ecolor="lightgray", capsize=3),
                )
            #
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim((0.5, 1))
            ax.set_xticks(x)
            ax.set_xticklabels(u)
            ax.set_title("{} - {}".format(discipline,population))

    # fig.legend()
    fig.tight_layout()
    return fig


def draw_corr_by_batch(output_dir_name):
    """
    correlations between rank scores derived by two independant batches of students
    """
    disciplines=["Physics","Chemistry","Ethics"]
    populations=["all","switchers"]

    fig,axs = plt.subplots(len(disciplines),len(populations),figsize=(12,9))

    for di,discipline in enumerate(disciplines):
        for p,population in enumerate(populations):

            ax=axs[di,p]
            output_dir=os.path.join(
                output_dir_name,
                discipline,
                population
            )

            results = []

            rank_score_types = [r for r in RANK_SCORE_TYPES if r != "wc"]
            for rank_score_type in rank_score_types:
                results_dir_discipline = os.path.join(
                    output_dir,
                    "model_fit",
                    rank_score_type,
                    "rank_scores_by_batch",
                )
                topics = os.listdir(results_dir_discipline)
                for topic in topics:
                    fp = os.path.join(results_dir_discipline, topic)
                    with open(fp, "r") as f:
                        rank_scores = json.load(f)

                    for transition in rank_scores:
                        d = {}
                        df_rank_scores = pd.DataFrame(rank_scores[transition])
                        try:
                            d["r"] = df_rank_scores.dropna().corr()["batch1"]["batch2"]
                            d["N"] = df_rank_scores.shape[0]
                            d["N_common"] = df_rank_scores.dropna().shape[0]
                            d["rank_score_type"] = RANK_SCORE_TYPES_RENAMED[rank_score_type]
                            d["topic"] = topic
                            d["transition"] = transition
                            results.append(d)
                        except KeyError:
                            pass
                            # print(topic)
                            # print(transition)
                            # print(df_rank_scores)
            df = pd.DataFrame(results)

            df_table = (
                df.dropna()
                .groupby(["rank_score_type", "transition"])
                .apply(lambda x: summary_batches_by_transition_corr(x))
                .reset_index()
            )

            df_table["rank_score_type"] = pd.Categorical(
                df_table["rank_score_type"],
                [RANK_SCORE_TYPES_RENAMED[r] for r in RANK_SCORE_TYPES if r != "wc"],
            )
            df_table = df_table.sort_values("rank_score_type")

            cat = "rank_score_type"
            subcat = "transition"
            val = "r"
            err = "std"
            ymin = 0
            u = df_table[cat].unique()
            x = np.arange(len(u))
            subx = df_table[subcat].unique()
            offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 4.0)
            width = np.diff(offsets).mean()

            plt.style.use("ggplot")

            for i, gr in enumerate(subx):
                dfg = df_table[df_table[subcat] == gr]
                ax.scatter(
                    x + offsets[i],
                    dfg[val].values,
                    marker="o",
                    label=TRANSITION_LABELS[discipline][gr],
                    color=TRANSITION_COLORS[discipline][gr],
                )
                ax.errorbar(
                    x=x + offsets[i],
                    y=dfg[val].values,
                    yerr=dfg[err].values / 2,
                    color=TRANSITION_COLORS[discipline][gr],
                    alpha=0.3,
                )
            ax.set_xlabel("Rank Score Type")
            ax.set_ylabel("Pearson Correlation")
            ax.set_ylim((ymin, 1))
            ax.set_xticks(x)
            ax.set_xticklabels(u)
            ax.set_title("{} - {}".format(discipline,population))
    fig.tight_layout()
    return fig


def draw_kendalltau_by_time(discipline,output_dir):
    """
    plot kendall tau between rankings at each normalized time step \sigma_t and
    \sigma_final
    """
    thresh = 0.05

    MAX_TIMESTEPS = 100
    rank_score_types = ["BT", "winrate", "winrate_no_pairs", "elo", "crowd_BT"]

    plt.style.use("ggplot")
    fig, axs = plt.subplots(figsize=(6, 4))
    topic_files = os.listdir(os.path.join(output_dir, "data"))
    topics = [t.replace(".csv", "") for t in topic_files]

    for r, rank_score_type in enumerate(rank_score_types):
        print("{}".format(rank_score_type))

        results_dir_discipline = os.path.join(
            output_dir, rank_score_type, "rankings_by_time"
        )
        X = np.empty((len(topic_files), MAX_TIMESTEPS))
        X[:] = np.nan

        for i, topic in enumerate(topics):
            # get rank scores
            fp = os.path.join(results_dir_discipline, "{}.json".format(topic))
            with open(fp, "r") as f:
                ranks_dict_all = json.load(f)

            # convert final rank scores to ranked list
            final_ranking = [
                k
                for k, v in sorted(
                    ranks_dict_all[-1].items(), key=lambda x: x[1], reverse=True
                )
            ]
            # scale to get ~100 time steps
            t_index = [
                math.floor(np.percentile(range(len(ranks_dict_all)), i))
                for i in range(MAX_TIMESTEPS)
            ]

            # compile ktau's
            for tp, t in enumerate(t_index):
                ranks_dict = ranks_dict_all[t]
                ranking = [
                    k
                    for k, v in sorted(
                        ranks_dict.items(), key=lambda x: x[1], reverse=True
                    )
                ]

                ranking_at_t = [a for a in ranking if a in final_ranking]
                final_ranking = [a for a in final_ranking if a in ranking_at_t]
                ktau, p = kendalltau(final_ranking, ranking_at_t)
                if p < thresh:
                    X[i, tp] = ktau

        means, stds = [], []
        for t in range(MAX_TIMESTEPS):
            m, s = X[:, t][~np.isnan(X[:, t])].mean(), X[:, t][~np.isnan(X[:, t])].std()
            means.append(m)
            stds.append(s)

        axs.plot(
            range(MAX_TIMESTEPS),
            means,
            label=RANK_SCORE_TYPES_RENAMED[rank_score_type],
            color=RANK_SCORE_TYPE_COLORS[rank_score_type],
        )
        axs.fill_between(
            x=range(MAX_TIMESTEPS),
            y1=np.array(means) - np.array(stds),
            y2=np.array(means) + np.array(stds),
            color="lightgray",
            alpha=0.4,
        )
    axs.legend()
    axs.set(
        xlabel="Percentile Rank of Time",
        ylabel=r"Kendall $\tau$ $\sigma_t$ with $\sigma_{final}$",
    )
    return fig


def main(
    figures: (
        "Which figures to remake",
        "positional",
        None,
        str,
        ["all", "corr_plot", "acc_by_transition", "corr_by_batch", "kendalltau_by_time"],
    ),
    discipline: (
        "Which discipline to consider",
        "positional",
        None,
        str,
        ["Ethics", "Physics","Chemistry"],
    ),
    output_dir_name: (
        "Directory name for results",
        "positional",
        None,
        str,
    ),
    filter_switchers: (
        "keep only students who switch their answer",
        "flag",
        "switch",
        bool,
    ),
):
    """
    generate plots and tables for LAK 2021 article
    """
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


    if filter_switchers:
        population="switchers"
        output_dir = os.path.join(data_loaders.BASE_DIR, "tmp", output_dir_name, discipline,population)
    else:
        population="all"
        output_dir = os.path.join(data_loaders.BASE_DIR, "tmp", output_dir_name, discipline,population)

    if figures == "all":
        print("summary of data by transition")
        df_summary_table = summary_table(discipline=discipline,output_dir=output_dir)
        fp = os.path.join(THESIS_DIR,"data","theme3", "df_summary_{}_{}.tex".format(discipline,population))
        df_summary_table.to_latex(fp)
        print(fp)

    if figures == "all" or figures == "corr_plot":
        ### Load data
        print("loading data for corr_plot")
        df, acc_trans, rank_scores = load_data_for_plots(discipline=discipline,output_dir=output_dir)

        ### draw corr_plot
        print("corr plot")
        if discipline == "Ethics":
            transition = None
        else:
            transition = "rr"

        fig = draw_corr_plot(rank_scores, discipline, transition=transition,output_dir=output_dir)
        fp = os.path.join(THESIS_DIR, "img","theme3", "corr_plot_{}_{}.pgf".format(discipline,population))
        print(fp)
        fig.savefig(fp)

    if figures == "all" or figures == "acc_by_transition":
        ### draw accuracies by transition for each rank_score_type
        fig = draw_acc_by_transition(output_dir_name=output_dir_name)
        fp = os.path.join(
            THESIS_DIR, "img","theme3", "acc_by_transition.pgf"
        )
        print(fp)
        fig.savefig(fp)

    if figures == "all" or figures == "corr_by_batch":
        ### Correlation between rank scores of independant batches of students
        #  by transition for each rank_score_type
        fig = draw_corr_by_batch(output_dir_name=output_dir_name)
        fp = os.path.join(THESIS_DIR, "img","theme3", "corr_by_batch.pgf")
        print(fp)
        fig.savefig(fp)

    # if figures == "all" or figures == "kendalltau_by_time":
    #     ### Correlation between rank scores of independant batches of students
    #     #  by transition for each rank_score_type
    #     fig = draw_kendalltau_by_time(discipline=discipline,output_dir=output_dir)
    #     fp = os.path.join(
    #         THESIS_DIR, "img","theme3", "kendalltau_by_time_{}_{}.pgf".format(discipline,population)
    #     )
    #     print(fp)
    #     fig.savefig(fp)


if __name__ == "__main__":
    import plac

    plac.call(main)
