import os
import json
import plac

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from scipy.stats import kendalltau, rankdata
import matplotlib.pyplot as plt
from itertools import combinations, permutations

import choix

import data_loaders

only_one_arg_pair = {}
model = "argBT"
results_dir = os.path.join(data_loaders.BASE_DIR, "tmp", model)


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


def build_rankings():
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

    results_acc = {}
    ground_truth_rankings = {}
    ties = {}

    for discipline in ["Ethics", "Physics", "Biology", "Chemistry"]:
        _, _, df_all = data_loaders.load_dalite_data(discipline=discipline)
        df_all = df_all[df_all["a1_id"] != df_all["a2_id"]]

        results_acc[discipline] = {}
        ground_truth_rankings[discipline] = {}
        ties[discipline] = {}
        for topic, df_topic in df_all.groupby("topic"):

            print(topic)

            results_acc[discipline][topic] = []
            ground_truth_rankings[discipline][topic] = []
            ties[discipline][topic] = pd.DataFrame()

            for counter, (r, df_r) in enumerate(
                df_topic.groupby("annotation_rank_by_time")
            ):
                if r % 10 == 0:
                    print(
                        "time step {}/{}".format(
                            counter,
                            df_topic["annotation_rank_by_time"].value_counts().shape[0],
                        )
                    )

                pairs_train = df_topic[df_topic["annotation_rank_by_time"] < r]
                pairs_test = df_topic[df_topic["annotation_rank_by_time"] == r].copy()
                if pairs_train.shape[0] > 0:
                    arg_ids_train = list(
                        set(
                            pd.concat(
                                [pairs_train["a1_id"], pairs_train["a2_id"]]
                            ).values
                        )
                    )

                    # choix requires integers as ids
                    arg_ids_dict_train = {s: i for i, s in enumerate(arg_ids_train)}
                    arg_ids_dict_train_reverse = {
                        v: k for k, v in arg_ids_dict_train.items()
                    }

                    # make tuples for choix
                    n_items = len(arg_ids_train)
                    data = []
                    for i, x in pairs_train.iterrows():
                        if x["label"] == "a1":
                            data.append(
                                (
                                    arg_ids_dict_train[x["a1_id"]],
                                    arg_ids_dict_train[x["a2_id"]],
                                )
                            )
                        else:
                            data.append(
                                (
                                    arg_ids_dict_train[x["a2_id"]],
                                    arg_ids_dict_train[x["a1_id"]],
                                )
                            )
                    # fit BT model
                    params = choix.ilsr_pairwise(
                        n_items, data, alpha=0.01
                    )  # smaller alpha leads to runtime error
                    # save ground truth sorted ranks
                    sorted_arg_ids = [
                        arg_ids_dict_train_reverse[p] for p in np.argsort(params)[::-1]
                    ]
                    ground_truth_rankings[discipline][topic].append(sorted_arg_ids)

                    ranks_dict = {
                        arg_id: param for arg_id, param in zip(arg_ids_train, params)
                    }

                    pairs_test["a1_rank"] = pairs_test["a1_id"].map(ranks_dict)
                    pairs_test["a2_rank"] = pairs_test["a2_id"].map(ranks_dict)

                    pairs_test_ = pairs_test.dropna().copy()

                    if pairs_test_.shape[0] > 0:
                        pairs_test_["label_pred"] = pairs_test_.apply(
                            lambda x: "a1"
                            if x["a1_rank"] > x["a2_rank"]
                            else ("a2" if x["a1_rank"] < x["a2_rank"] else np.nan),
                            axis=1,
                        )
                        pairs_test_ = pairs_test_.dropna().copy()

                        try:
                            results_acc[discipline][topic].append(
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
                            ties[discipline][topic].append(
                                pairs_test_[
                                    [
                                        "#id",
                                        "a1_rank",
                                        "a2_rank",
                                        "annotation_rank_by_time",
                                    ]
                                ]
                            )
                            pairs_test_ = pairs_test_.dropna().copy()
                            if pairs_test_.shape[0] > 0:
                                pairs_test_["label_pred"] = pairs_test_.apply(
                                    lambda x: "a1"
                                    if x["a1_rank"] > x["a2_rank"]
                                    else (
                                        "a2" if x["a1_rank"] < x["a2_rank"] else np.nan
                                    ),
                                    axis=1,
                                )
                                results_acc[discipline][topic].append(
                                    {
                                        "r": r,
                                        "n": pairs_test_.shape[0],
                                        "acc": accuracy_score(
                                            y_true=pairs_test_["label"],
                                            y_pred=pairs_test_["label_pred"],
                                        ),
                                    }
                                )

            fp = os.path.join(results_dir, discipline,"accuracies", "{}.json".format(topic))
            with open(fp, "w+") as f:
                f.write(json.dumps(results_acc[discipline][topic]))

            fp=os.path.join(results_dir,discipline,"rankings","{}.json".format(topic))
            with open(fp,"w+") as f:
                f.write(json.dumps(ground_truth_rankings[discipline][topic]))
    return


if __name__ == "__main__":
    import plac

    plac.call(build_rankings)
