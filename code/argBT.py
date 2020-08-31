import os
import json
import plac

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from scipy.stats import kendalltau, rankdata
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

import choix

import data_loaders
from make_pairs import get_ethics_answers,get_mydalite_answers

only_one_arg_pair = {}
model = "argBT"
RESULTS_DIR = os.path.join(data_loaders.BASE_DIR, "tmp", "fine_grained_arg_rankings")
MAX_ITERATIONS = 1500


# http://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html
def kendalltau_dist(rank_a, rank_b):
    tau = 0
    n_candidates = len(rank_a)
    for i, j in combinations(range(n_candidates), 2):
        tau += np.sign(rank_a[i] - rank_a[j]) == -np.sign(rank_b[i] - rank_b[j])
    return tau


def kendalltau_dist_norm(rank_a, rank_b):
    tau = kendalltau_dist(rank_a, rank_b)
    n_items = len(rank_a)
    return 2 * tau / (n_items * (n_items - 1))


def get_rankings(pairs_train):
    """
    Arguments:
    ----------
        - pairs_train: pandas DataFrame, with columns
            - a1_id
            - a2_id
            - label: which of the columns, "a1" or "a2" is the winning argument

    Returns:
    --------
        -
    """
    arg_ids_train = list(
        set(pd.concat([pairs_train["a1_id"], pairs_train["a2_id"],]).values)
    )
    # choix requires integers as ids
    arg_ids_dict_train = {s: i for i, s in enumerate(arg_ids_train)}
    arg_ids_dict_train_reverse = {v: k for k, v in arg_ids_dict_train.items()}

    # make tuples for choix
    n_items = len(arg_ids_train)
    data = []
    for i, x in pairs_train.iterrows():
        if x["label"] == "a1":
            data.append(
                (arg_ids_dict_train[x["a1_id"]], arg_ids_dict_train[x["a2_id"]],)
            )
        else:
            data.append(
                (arg_ids_dict_train[x["a2_id"]], arg_ids_dict_train[x["a1_id"]],)
            )
    # fit BT model
    params = choix.ilsr_pairwise(
        n_items, data, alpha=0.01
    )  # smaller alpha leads to runtime error
    # save ground truth sorted ranks
    sorted_arg_ids = [arg_ids_dict_train_reverse[p] for p in np.argsort(params)[::-1]]

    ranks_dict = {arg_id: param for arg_id, param in zip(arg_ids_train, params)}

    return sorted_arg_ids, ranks_dict


def get_rankings_baseline(df_train):
    """
    Arguments:
    ----------
        - df_train: pandas DataFrame with column "rationales" (what was shown)
        and "chosen_rationale_id", for what was chosen
    Returns:
    --------
        - sorted_arg_ids
        - ranks_dict
    """
    times_shown_counter = Counter()
    s=df_train["rationales"].dropna().apply(lambda x: [int(k) for k in x.strip("[]").replace(" ","").split(",") if k!=""])
    _ = s.apply(lambda x: times_shown_counter.update(x))

    votes_count=df_train["chosen_rationale_id"].value_counts().to_dict()
    ranks_dict={
        "arg{}".format(k):(v+1)/(times_shown_counter[k]+8) for k,v in votes_count.items()
    }
    sorted_arg_ids=["arg{}".format(p[0]) for p in sorted(ranks_dict.items(),key=lambda x: x[1])[::-1]]
    return sorted_arg_ids,ranks_dict

def pairwise_predict(x):

    if x["a1_rank"] > x["a2_rank"]:
        pred = "a1"
    elif x["a1_rank"] < x["a2_rank"]:
        pred = "a2"
    else:
        pred = np.nan

    return pred

def get_data_dir(discipline):
    return os.path.join(RESULTS_DIR, discipline, "data")

def build_rankings_by_topic(topic,discipline,rank_score_type):
    """
    """
    data_dir_discipline = get_data_dir(discipline)

    fp = os.path.join(data_dir_discipline,"{}.csv".format(topic))
    df_topic = pd.read_csv(fp)
    df_topic = df_topic[~df_topic["user_token"].isna()].sort_values("a_rank_by_time")
    # load pairs
    pairs_df = pd.read_csv(
        os.path.join(
            "{}_pairs".format(data_dir_discipline),
            "pairs_{}.csv".format(topic)
        )
    )
    pairs_df = pairs_df[pairs_df["a1_id"] != pairs_df["a2_id"]]

    accuracies = []
    ground_truth_rankings = []
    rankings_by_batch = []

    for counter, (r, df_r) in enumerate(
        pairs_df.groupby("annotation_rank_by_time")
    ):

        if r <= MAX_ITERATIONS:

            if r % 10 == 0:
                print(
                    "\t\ttime step {}/{}".format(
                        counter,
                        pairs_df["annotation_rank_by_time"].value_counts().shape[0],
                    )
                )

            pairs_train = pairs_df[pairs_df["annotation_rank_by_time"] < r]
            pairs_test = pairs_df[pairs_df["annotation_rank_by_time"] == r
            ].copy()

            df_train=df_topic[df_topic["a_rank_by_time"]<r]
            students = df_train["user_token"].to_list()

            if pairs_train.shape[0] > 0 and len(students) > 10:

                if rank_score_type=="baseline":
                    # learn rankings from all previous students
                    sorted_arg_ids, ranks_dict = get_rankings_baseline(
                        df_train=df_train
                    )
                else:
                    # learn rankings from all previous students
                    sorted_arg_ids, ranks_dict = get_rankings(
                        pairs_train=pairs_train
                    )

                ground_truth_rankings.append(sorted_arg_ids)

                # test ability of rankings to predict winning argument in
                # held out pairs at current timestep
                pairs_test["a1_rank"] = pairs_test["a1_id"].map(ranks_dict)
                pairs_test["a2_rank"] = pairs_test["a2_id"].map(ranks_dict)

                # pairs with current student's argument must be dropped
                # as it is as yet unseen
                pairs_test_ = (
                    pairs_test[["a1_rank", "a2_rank", "label"]].dropna().copy()
                )

                # for each pair in held out pairs at current timestep,
                # predict winner based on higher BT param
                if pairs_test_.shape[0] > 0:
                    pairs_test_["label_pred"] = pairs_test_.apply(
                        lambda x: pairwise_predict(x), axis=1,
                    )
                    pairs_test_ = pairs_test_.dropna().copy()
                    try:
                        accuracies.append(
                            {
                                "r": r,
                                "n": pairs_test_.shape[0],
                                "acc": accuracy_score(
                                    y_true=pairs_test_["label"],
                                    y_pred=pairs_test_["label_pred"],
                                ),
                            }
                        )
                    except TypeError:
                        # drop any ties
                        n_ties = pairs_test_.isna().shape[0]
                        pairs_test_ = pairs_test_.dropna().copy()

                        if pairs_test_.shape[0] > 0:
                            pairs_test_["label_pred"] = pairs_test_.apply(
                                lambda x: pairwise_predict(x), axis=1,
                            )
                            accuracies.append(
                                {
                                    "r": r,
                                    "n": pairs_test_.shape[0],
                                    "acc": accuracy_score(
                                        y_true=pairs_test_["label"],
                                        y_pred=pairs_test_["label_pred"],
                                    ),
                                    "ties": n_ties,
                                }
                            )
                # make two batches of students, interleaved in time
                student_batch1 = students[::2]
                student_batch2 = [
                    s for s in students if s not in student_batch1
                ]

                # get rankings for each batch and save
                batch_rankings = {}
                for sb, student_batch in zip(
                    ["batch1", "batch2"], [student_batch1, student_batch2]
                ):

                    if rank_score_type=="baseline":
                        df_train_batch = df_topic[df_topic["user_token"].isin(student_batch)]
                        sorted_arg_ids, ranks_dict = get_rankings_baseline(
                            df_train=df_train_batch
                        )
                    else:
                        pairs_train_batch = pairs_train[
                            pairs_train["annotator"].isin(student_batch)
                        ]
                        sorted_arg_ids, ranks_dict = get_rankings(
                            pairs_train=pairs_train_batch
                        )

                    batch_rankings[sb] = sorted_arg_ids

                rankings_by_batch.append(batch_rankings)
    return accuracies, ground_truth_rankings, rankings_by_batch

def build_rankings(discipline, rank_score_type="baseline"):
    """
    - load dalite arg pairs by discipline
    - for each topic:
        - iterate through each timestep, which represents the pairs of arguments
        a student if presented with, labelled by the one that is chosen as
        more convincing
        - take all pairs leading up to that time step, and learn Bradley-Terry
        parameters (argument strength ratings).
        - use these to predict which argument is most convincing amongst the
        pairs for the current timestep. (only possible for previously seen
        arguments)
        - save strength ratings of all argument rankings at each timestep,
        as well as classification accuracy
    """

    print(discipline)

    data_dir_discipline = get_data_dir(discipline)
    results_dir_discipline = os.path.join(RESULTS_DIR, discipline,rank_score_type)

    # make results directories if they do not exist:
    if not os.path.exists(results_dir_discipline):
        os.mkdir(results_dir_discipline)
        os.mkdir(
            os.path.join(
                results_dir_discipline,
                "accuracies",
            )
        )
        os.mkdir(
            os.path.join(
                results_dir_discipline,
                "rankings",
            )
        )
        os.mkdir(
            os.path.join(
                results_dir_discipline,
                "rankings_by_batch",
            )
        )

    # sort files by size to get biggest ones done first
    # https://stackoverflow.com/a/20253803
    all_files = (
        os.path.join(data_dir_discipline, filename)
        for basedir, dirs, files in os.walk(data_dir_discipline)
        for filename in files
    )
    all_files = sorted(all_files, key=os.path.getsize, reverse=True)

    topics = [os.path.basename(fp)[:-4] for fp in all_files]

    topics_already_done = [
        t[:-5]
        for t in os.listdir(
            os.path.join(
                results_dir_discipline,
                "rankings_by_batch",
            )
        )
    ]

    topics = [t for t in topics if t not in topics_already_done]

    for i,topic in enumerate(topics):
        print("\t{}/{} {}".format(i,len(topics),topic))

        accuracies, ground_truth_rankings, rankings_by_batch = build_rankings_by_topic(
            topic=topic,
            discipline=discipline,
            rank_score_type=rank_score_type
            )

        fp = os.path.join(
            results_dir_discipline,"accuracies", "{}.json".format(topic)
        )
        with open(fp, "w+") as f:
            f.write(json.dumps(accuracies, indent=2))

        fp = os.path.join(
            results_dir_discipline,"rankings", "{}.json".format(topic)
        )
        with open(fp, "w+") as f:
            f.write(json.dumps(ground_truth_rankings, indent=2))

        fp = os.path.join(
            results_dir_discipline,"rankings_by_batch", "{}.json".format(topic),
        )
        with open(fp, "w+") as f:
            f.write(json.dumps(rankings_by_batch, indent=2))

    return


if __name__ == "__main__":
    import plac

    plac.call(build_rankings)
