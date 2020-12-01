import os
import json
import plac
import pandas as pd
from data_loaders import BASE_DIR
from argBT import get_topic_data,pairwise_predict
from sklearn.metrics import accuracy_score

def main(
    discipline: (
        "Discipline",
        "positional",
        None,
        str,
        ["Physics", "Chemistry", "Ethics"],
    ),
    rank_score_type: (
        "Rank Score Type",
        "positional",
        None,
        str,
        [
            "BT",
            "elo",
            "crowd_BT",
            "crowdBT_filtered",
            "winrate",
            "winrate_no_pairs",
            "wc",
        ],
    ),
    output_dir_name: ("Directory name for results", "positional", None, str,),
):
    population="all"
    output_dir=os.path.join(
        BASE_DIR,"tmp",output_dir_name,discipline,population,"time_series",rank_score_type,"rankings_by_time"
    )
    filenames=os.listdir(output_dir)
    results = []
    for t,fn in enumerate(filenames):
        if t%10==0:
            print(f"\t{t}/{len(filenames)}")
        fp=os.path.join(output_dir,fn)
        with open(fp,"r") as f:
            rankings_by_time=json.load(f)

        topic=fn[:-5]
        data_dir=os.path.join(
            BASE_DIR,"tmp",output_dir_name,discipline,population
        )
        pairs_df,df_topic=get_topic_data(output_dir=data_dir,topic=topic,discipline=discipline)

        # print(f"{len(rankings_by_time)} ranking-times-steps; {pairs_df.shape[0]} pairs")
        for i,ranks_dict in enumerate(rankings_by_time):
            pairs_test=pairs_df[(
                (pairs_df["a1_id"].isin(ranks_dict))&(pairs_df["a2_id"].isin(ranks_dict))
            )].copy()

            # get model fit on training data at current timestep
            pairs_test["a1_rank"] = pairs_test["a1_id"].map(ranks_dict)
            pairs_test["a2_rank"] = pairs_test["a2_id"].map(ranks_dict)

            for transition,pairs_test_transition_ in pairs_test.groupby("transition"):
                d={}
                pairs_test_transition=pairs_test_transition_.copy()
                pairs_test_transition["label_pred"] = pairs_test.apply(
                    lambda x: pairwise_predict(x), axis=1,
                )
                d["N_ties"]=pairs_test_transition["label_pred"].isna().sum()

                pairs_test_transition_filtered=pairs_test_transition.dropna(subset=["label_pred"])

                d["acc"]=accuracy_score(
                            y_true=pairs_test_transition_filtered["label"],
                            y_pred=pairs_test_transition_filtered["label_pred"],
                        )
                d["transition"]=transition
                d["N_test"]=pairs_test_transition.shape[0]
                d["i"]=str(i)
                d["topic"]=topic
                d["rank_score_type"]=rank_score_type
                results.append(d)

    fp=os.path.join(output_dir,os.pardir,f"accuracies_by_time_{discipline}_{rank_score_type}.csv")

    pd.DataFrame(results).to_csv(fp)

if __name__ == '__main__':
    plac.call(main)
