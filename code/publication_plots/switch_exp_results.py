import os,json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score,accuracy_score


import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTICLE_DIR = os.path.join(BASE_DIR,"article","lak2021")

#careful if dir names change
RESULTS_DIR = os.path.join(BASE_DIR,"tmp","fine_grained_arg_rankings")

def switch_exp_summary(x):
    d={}
    d["n"]=x["timestep"].max()
    d["acc_LR"]=np.round(
        accuracy_score(
            y_true=x["y_true"],
            y_pred=x["prediction_LR"]
        ),4
    )
    d["acc_RF"]=np.round(
        accuracy_score(
            y_true=x["y_true"],
            y_pred=x["prediction_RF"]
        ),4
    )
    d["f1_LR"]=np.round(
        f1_score(
            y_true=x["y_true"],
            y_pred=x["prediction_LR"]
        ),4
    )
    d["f1_RF"]=np.round(
        f1_score(
            y_true=x["y_true"],
            y_pred=x["prediction_RF"]
        ),4
    )

    return pd.Series(d,index=["n","acc_LR","acc_RF","f1_LR","f1_RF"])


def summary_by_disc(x):
    d={}
    d["n"]="{:0.0f}".format(x["n"].sum())
    d["acc_LR"]="{:0.2f} ({:0.2f})".format(
        np.average(x["acc_LR"],weights=x["n"]),
        np.std(x["acc_LR"])
    )
    d["acc_RF"]="{:0.2f} ({:0.2f})".format(
        np.average(x["acc_RF"],weights=x["n"]),
        np.std(x["acc_RF"])
    )
    return(pd.Series(d,index=["n","acc_LR","acc_RF"]))




def main():
    disciplines=["Physics","Chemistry","Biology"]

    ###########
    # accuracy
    results=[]
    cols=["topic","timestep","test_answer_id","n","y_true"]
    cols_pred=["prediction_LR","prediction_RF"]

    for discipline in disciplines:
        results_dir_discipline = os.path.join(RESULTS_DIR,discipline,"results")
        topics=os.listdir(results_dir_discipline)
        for topic in topics:
            fp=os.path.join(results_dir_discipline,topic)
            with open(fp,"r") as f:
                d=json.load(f)
            for dt in d:
                if dt:
                    dtf={c:dt[c] for c in cols}
                    for c in cols_pred:
                        dtf.update({c:dt[c][0]})
                    dtf.update({"discipline":discipline})
                    results.append(dtf)
    df=pd.DataFrame(results)

    df_d=df.groupby(["discipline","topic"]).apply(lambda x: switch_exp_summary(x))


    df_final=df_d.groupby("discipline").apply(lambda x: summary_by_disc(x))

    fp=os.path.join(ARTICLE_DIR,"data","switch_exp_acc.tex")
    df_final.to_latex(fp)

    #####################
    # feature importances
    results_f=[]
    cols=["topic","timestep","n"]
    cols_features=["feature_names_LR","feature_names_RF"]
    disciplines=["Biology","Chemistry","Physics"]
    for discipline in disciplines:
        results_dir_discipline = os.path.join(RESULTS_DIR,discipline,"results")
        topics=os.listdir(results_dir_discipline)
        for topic in topics:
            fp=os.path.join(results_dir_discipline,topic)
            with open(fp,"r") as f:
                d=json.load(f)
            for dt in d:
                if dt:
                    for col_feature in cols_features:
                        dtfa={
                            "topic":topic,
                            "model":col_feature[-2:],
                            "timestep":dt["timestep"],
                            "pred_true":bool(
                                dt["prediction_{}".format(col_feature[-2:])][0]==dt["y_true"]
                            )
                        }
                        dtf=dt[col_feature]
                        for weight,feature_name in dtf:
                            dtfa.update({feature_name:weight})
                        dtfa.update({"discipline":discipline})
                        results_f.append(dtfa)
    dff=pd.DataFrame(results_f)
    # dff.groupby("discipline").size()
    cols_features=['rationale_word_count',
           'shown_rationale_word_count_mean', 'shown_convincingness_BT_mean',
           'shown_convincingness_baseline_mean', 'shown_rationale_word_count_max',
           'shown_convincingness_BT_max', 'shown_convincingness_baseline_max',
           'shown_rationale_word_count_min', 'shown_convincingness_BT_min',
           'shown_convincingness_baseline_min', 'n_shown_short',
           'n_shown_shorter_than_own', 'n_shown_longer_than_own', 'first_correct']

    dfrf=dff[dff["model"]=="RF"].copy()

    means=dfrf[cols_features].mean()
    stds=dfrf[cols_features].std()

    df_means=means.to_frame().rename(columns={0:"mean"}).join(
        stds.to_frame().rename(columns={0:"std"})
    ).sort_values("mean",ascending=False)

    ftypes=[]
    for f in list(df_means.index):
        if "baseline" in f:
            ftypes.append("WinRate")
        elif "BT" in f:
            ftypes.append("BT")
        else:
            ftypes.append("Surface")
    df_means["type"]=ftypes


    df_means.index=df_means.index\
    .str.replace("shown_","")\
    .str.replace("convincingness_","")\
    .str.replace("baseline","WinRate")\
    .str.replace("rationale_","")\
    .str.replace("word_count","WC")

    colors={
        "Surface":"lightgrey",
        "WinRate":"red",
        "BT":"royalblue"
    }

    df_means["color"]=df_means["type"].map(colors)

    # df_means
    plt.bar(
        x=df_means.index,
        height=df_means["mean"],
        yerr=df_means["std"],
        color=df_means["color"],
        error_kw=dict(ecolor='gray', lw=1, capsize=2, capthick=1)
    )
    plt.xticks(range(df_means.shape[0]),df_means.index,rotation=75)
    plt.ylabel("Average Feature Importance in Random Forest")
    fp=os.path.join(ARTICLE_DIR,"img","switch_exp_RF.pgf")
    plt.savefig(fp)

if __name__ == "__main__":
    import plac

    plac.call(main)
