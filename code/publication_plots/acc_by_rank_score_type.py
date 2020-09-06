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
RESULTS_DIR = os.path.join(BASE_DIR,"tmp","fine_grained_arg_rankings")

def main():
    disciplines=["Physics","Chemistry","Biology"]
    ground_truth_rankings={}
    rankings_by_batch={}

    df=pd.DataFrame()

    for rank_score_type in ["baseline","BT"]:
    #     print("{}".format(rank_score_type))
        df_all_disciplines=pd.DataFrame()

        for discipline in disciplines:
    #         print("\t{}".format(discipline))

            df_acc_disc=pd.DataFrame()

            results_dir_discipline = os.path.join(
                RESULTS_DIR,
                discipline,
                rank_score_type,
            )
            topics=os.listdir(
                os.path.join(
                    results_dir_discipline,
                    "accuracies"
                )
            )

            ground_truth_rankings[discipline]={}
            rankings_by_batch[discipline]={}
            for topic in topics:
                topic_csv = topic.replace(".json",".csv")

                fp=os.path.join(results_dir_discipline,"accuracies",topic)
                with open(fp,"r") as f:
                    df_acc_t=pd.DataFrame(json.load(f))

                df_acc_t["topic"]=topic

                df_acc_disc=pd.concat([df_acc_disc,df_acc_t])

            df_acc_disc["discipline"]=discipline
            df_all_disciplines=pd.concat([df_all_disciplines,df_acc_disc])

        df_all_disciplines["rank_score_type"]=rank_score_type

        df=pd.concat([df,df_all_disciplines])

    common_cols=["r","topic","discipline","acc","n"]
    merge_cols=["r","topic","discipline"]
    df2=pd.merge(
        df.loc[df["rank_score_type"]=="baseline",common_cols].rename(
            columns={
                "acc":"acc_WinRate",
                "n":"n_WinRate"
            }),
        df.loc[df["rank_score_type"]=="BT",common_cols].rename(
            columns={
                "acc":"acc_BT",
                "n":"n_BT"
            }),
        on=merge_cols
    )
    df3=df2.melt(id_vars=merge_cols,value_vars=["acc_WinRate","acc_BT"]).rename(columns={"value":"acc"})
    df3["rank_score_type"]=df3["variable"].str[4:]
    sns.set()
    ax=sns.pointplot(x="rank_score_type",y="acc",data=df3,hue="discipline")
    ax.set_xticklabels(["Win Rate","Bradley-Terry"])
    ax.set_ylabel("Average pairwise classification accuracy")
    ax.set_xlabel("Convincingness Score Type")
    ax.set_ylim((0.5,1))

    fp=os.path.join(BASE_DIR,"article","lak2021","img","acc_by_rank_score_type.pgf")
    print(fp)
    plt.savefig(fp)


if __name__ == "__main__":
    import plac

    plac.call(main)
