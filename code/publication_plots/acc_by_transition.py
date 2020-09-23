import os,json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#careful if dir names change
RESULTS_DIR = os.path.join(BASE_DIR,"tmp","measure_convincingness")


def my_summary(x):
    d={}
    d["N"] = "{:0.0f}".format(x["n"].sum())
    d["acc"] = np.average(x["acc"],weights=x["n"])
    d["std"] = np.std(x["acc"])

    return(pd.Series(d,index=["N","acc","std"]))


def main():

    ### Load data
    print("loading data")
    discipline="Physics"
    df=pd.DataFrame()
    rank_scores,acc_trans={},{}
    rank_score_types = ["wc","winrate","elo","crowd_BT","BT"]
    for rank_score_type in rank_score_types:
        print("\t{}".format(rank_score_type))
        results_dir_discipline = os.path.join(
            RESULTS_DIR,
            discipline,
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
        for topic in topics:

            fp=os.path.join(
                results_dir_discipline,
                "accuracies_train",
                "{}".format(topic)
            )
            with open(fp,"r") as f:
                d=json.load(f)

            df_acc_t=pd.DataFrame([{k:v for k,v in d1.items() if k!="acc_by_transition"} for d1 in d])
            df_acc_t["transition"]="all"

            df_acc_t["topic"]=topic
            df_acc=pd.concat([df_acc,df_acc_t])

            acc_trans[rank_score_type][topic]=[{k:v for k,v in d1.items() if k=="acc_by_transition"} for d1 in d]

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
    df["t"]=df.groupby(["topic","rank_score_type"])["r"].transform(lambda x: (x.rank(pct=True)*100).astype(int))

    # collect rank_scores
    df_all=pd.DataFrame()
    for rank_score_type in rank_score_types:
        df_all_topics=pd.DataFrame()
        for topic,scores in rank_scores[rank_score_type].items():
            df_scores=pd.DataFrame.from_dict(
                rank_scores[rank_score_type][topic][-1],
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
    df_pivot=pd.pivot(df_all,index="arg_id",columns="rank_score_type")["value"]
    topics = pd.pivot(df_all,index="arg_id",columns="rank_score_type")["topic"]
    df_pivot=pd.merge(
        df_pivot,
        topics[["wc"]].rename(columns={"wc":"topic"}),
        left_index=True,
        right_index=True
    )

    # append trantsition type for each arg
    print("loading transition data for each arg")
    data_dir = os.path.join(RESULTS_DIR,discipline,"data")
    df_topics=pd.DataFrame()
    for topic,df_pivot_topic in df_pivot.groupby("topic"):
        fp=os.path.join(data_dir,"{}.csv".format(topic))
        df_topic=pd.read_csv(fp)
        df_topic["arg_id"]="arg"+df_topic["id"].astype(str)
        df_topic=df_topic.set_index("arg_id")[["transition"]]
        df_topics = pd.concat([df_topics,df_topic])

    df_pivot["transition"]=df_pivot.index.map(df_topics["transition"].to_dict())

    # correlation plot for each transition type
    transitions = ["rr","rw","wr","ww"]
    transition_labels = {
        "rr":"Right -> Right",
        "rw":"Right -> Wrong",
        "wr":"Wrong -> Right",
        "ww":"Wrong -> Wrong",
    }
    fig,axs = plt.subplots(2,2,figsize=(11,9))
    cmap=sns.diverging_palette(230,20,as_cmap=True)
    for ax,transition in zip(axs.flatten(),transitions):
        corr=df_pivot.groupby("transition").corr().loc[transition]
        mask = np.triu(np.ones_like(corr,dtype=bool))
        sns.heatmap(
            corr.iloc[1:,:-1],
            mask=mask[1:,:-1],
            cmap=cmap,
            vmax=1.0,
            vmin=0.0,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"shrink":0.5},
        )
        ax.text(2.25,0.75,transition_labels[transition],size=14)
    for ax in axs.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.tight_layout()
    fp=os.path.join(BASE_DIR,"article","lak2021","img","corr_plot.pgf")
    print(fp)
    fig.savefig(fp)
    ### very very slow. load saved version into p as shortcut
    # p=pd.DataFrame()
    # for rank_score_type in rank_score_types:
    #     print(rank_score_type)
    #     for i,(topic,d_array) in enumerate(acc_trans[rank_score_type].items()):
    #         print("\t{}".format(i))
    #         print("\t\t{}".format(len(d_array)))
    #         for d in d_array:
    #             q=pd.DataFrame(d["acc_by_transition"])
    #             q["rank_score_type"]=rank_score_type
    #             q["topic"]=topic
    #             p=pd.concat([p,q])
    # df2=pd.concat([df,p.rename(columns={"n_shape":"n"})])
    print("loading df2")
    fp=os.path.join(RESULTS_DIR,discipline,"df2.csv")
    df2=pd.read_csv(fp)

    #### accuracies for each rank score type by transition
    df_plot=df2.dropna(subset=["acc"]).groupby(["rank_score_type","transition"]).apply(
        lambda x: my_summary(x)
    ).reset_index()
    df_plot.loc[df_plot["rank_score_type"]=="WordCount","rank_score_type"]="wc"

    col_ordering=["crowd_BT","BT","elo","winrate","wc"]
    transition_colors={
        "all":"gray",
        "rr":"green",
        "rw":"yellow",
        "wr":"blue",
        "ww":"red"
    }
    fig,ax=plt.subplots()
    sns.set()
    for transition,df_transition in df_plot.groupby("transition"):
        df_t=df_transition.set_index("rank_score_type").reindex(col_ordering)
        plt.plot(
            df_t.index,
            df_t["acc"],
            marker='o',
            label=transition,
            alpha=0.5,
            color=transition_colors[transition]
        )
        plt.errorbar(
            x=df_t.index,
            y=df_t["acc"],
            yerr=df_t["std"],
            alpha=0.5,
        )
    plt.legend()
    fp=os.path.join(BASE_DIR,"article","lak2021","img","acc_by_transition.pgf")
    print(fp)
    fig.savefig(fp)


if __name__ == "__main__":
    import plac

    plac.call(main)
