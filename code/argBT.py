import os
import json
import plac
import math
from pathlib import Path
import numpy as np
import pandas as pd

import spacy

nlp = spacy.load("en_core_web_sm")

from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import kendalltau, rankdata
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter, defaultdict

import choix

import data_loaders
from make_pairs import get_ethics_answers, get_mydalite_answers
from elo import get_initial_ratings, compute_updated_ratings
import crowd_bt

PRIORS = {
    "ALPHA": crowd_bt.ALPHA_PRIOR,
    "BETA": crowd_bt.BETA_PRIOR,
    "MU": crowd_bt.MU_PRIOR,
    "SIGMA_SQ": crowd_bt.SIGMA_SQ_PRIOR,
}

DROPPED_POS = ["PUNCT", "SPACE"]


def prior_factory(key):
    return lambda: PRIORS[key]


only_one_arg_pair = {}
model = "argBT"
RESULTS_DIR = os.path.join(data_loaders.BASE_DIR, "tmp", "measure_convincingness")
MAX_ITERATIONS = 1500
MIN_WORD_COUNT_DIFF = 5


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


def get_rankings_wc(df_train):
    """
    ranking = Word Count
    """

    rationales = df_train[["rationale", "id"]].values
    ranks_dict = {
        "arg{}".format(arg_id): len(
            [token for token in doc if token.pos not in DROPPED_POS]
        )
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }
    sorted_arg_ids = [
        k for k, v in sorted(ranks_dict.items(), key=lambda x: x[1], reverse=True)
    ]

    return sorted_arg_ids, ranks_dict


def get_rankings_crowdBT(pairs_train):
    """
    Arguments:
    ----------
        - pairs_train: pandas DataFrame, with columns
            - a1_id
            - a2_id
            - label: which of the columns, "a1" or "a2" is the winning argument

    Returns:
    --------
        - sorted_arg_ids - > list
        - ranks_dict
    """

    a1_winners = pairs_train.loc[pairs_train["label"] == "a1", :].rename(
        columns={"a1_id": "winner", "a2_id": "loser"}
    )
    a2_winners = pairs_train.loc[pairs_train["label"] == "a2", :].rename(
        columns={"a2_id": "winner", "a1_id": "loser"}
    )
    a = pd.concat([a1_winners, a2_winners])

    (alpha, beta, mu, sigma_sq,) = (
        defaultdict(prior_factory("ALPHA")),
        defaultdict(prior_factory("BETA")),
        defaultdict(prior_factory("MU")),
        defaultdict(prior_factory("SIGMA_SQ")),
    )

    for i, row in a.iterrows():
        (
            alpha[row["annotator"]],
            beta[row["annotator"]],
            mu[row["winner"]],
            sigma_sq[row["winner"]],
            mu[row["loser"]],
            sigma_sq[row["loser"]],
        ) = crowd_bt.update(
            alpha[row["annotator"]],
            beta[row["annotator"]],
            mu[row["winner"]],
            sigma_sq[row["winner"]],
            mu[row["loser"]],
            sigma_sq[row["loser"]],
        )

    ranks_dict = mu

    sorted_arg_ids = [
        k for k, v in sorted(ranks_dict.items(), key=lambda x: x[1], reverse=True)
    ]

    annotator_params = {
        annotator: beta_dist.mean(alpha[annotator], beta[annotator])
        for annotator in alpha.keys()
    }

    return sorted_arg_ids, ranks_dict, annotator_params


def get_rankings_BT(pairs_train):
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


def get_rankings_elo(pairs_df):
    """
    calculate elo ratings given labelled pairs
    """
    args = list(set(list(pairs_df["a1_id"]) + list(pairs_df["a2_id"])))
    items = get_initial_ratings(args)
    pairs_df["elo_outcome"] = pairs_df["label"].map({"a1": 1, "a2": 0})
    results = {
        (a1, a2): r for a1, a2, r in pairs_df[["a1_id", "a2_id", "elo_outcome"]].values
    }
    ranks_dict = compute_updated_ratings(items, results)

    sorted_arg_ids = [
        k for k, v in sorted(ranks_dict.items(), key=lambda x: x[1], reverse=True)
    ]

    return sorted_arg_ids, ranks_dict


def get_rankings_winrate(df_train):
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

    MIN_VOTES, MIN_SHOWN = 1, 8

    times_shown_counter = Counter()
    s = (
        df_train["rationales"]
        .dropna()
        .apply(
            lambda x: [
                int(k) for k in x.strip("[]").replace(" ", "").split(",") if k != ""
            ]
        )
    )
    _ = s.apply(lambda x: times_shown_counter.update(x))

    votes_count = df_train["chosen_rationale_id"].value_counts().to_dict()
    ranks_dict = {
        "arg{}".format(k): (v + MIN_VOTES) / (times_shown_counter[k] + MIN_SHOWN)
        for k, v in votes_count.items()
    }
    # need to add entries for rationales never shown
    never_chosen = [r for r in df_train["id"].to_list() if r not in votes_count]
    ranks_dict.update({"arg{}".format(r): MIN_VOTES / MIN_SHOWN for r in never_chosen})
    sorted_arg_ids = [
        "arg{}".format(p[0])
        for p in sorted(ranks_dict.items(), key=lambda x: x[1])[::-1]
    ]
    return sorted_arg_ids, ranks_dict


def pairwise_predict(x):

    if x["a1_rank"] > x["a2_rank"]:
        pred = "a1"
    elif x["a1_rank"] < x["a2_rank"]:
        pred = "a2"
    else:
        pred = np.nan  # "a{}".format(np.random.randint(low=1, high=3))

    return pred


def get_data_dir(discipline):
    return os.path.join(RESULTS_DIR, discipline, "data")


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


def get_ranking_model_fit(pairs_train, df_train, rank_score_type):
    """
    calculate rankings based on specified type, and run pairwise prediction
    on training pairs
    """

    if rank_score_type == "winrate":
        # ranking = times_chosen/times_shown
        sorted_arg_ids, ranks_dict = get_rankings_winrate(df_train=df_train)

    elif rank_score_type == "wc":
        # ranking = num tokens
        sorted_arg_ids, ranks_dict = get_rankings_wc(df_train=df_train)

    elif rank_score_type == "crowd_BT":
        # ranking + annotator params based on crowd_BT
        sorted_arg_ids, ranks_dict, annotator_params = get_rankings_crowdBT(
            pairs_train=pairs_train
        )

    elif rank_score_type == "elo":
        # ranking based on ELO match-ups, mean = 1500
        sorted_arg_ids, ranks_dict = get_rankings_elo(pairs_df=pairs_train)

    elif rank_score_type == "BT":
        # rankings from conventional Bradley Terry
        sorted_arg_ids, ranks_dict = get_rankings_BT(pairs_train=pairs_train)

    # get model fit on training data at current timestep
    pairs_train["a1_rank"] = pairs_train["a1_id"].map(ranks_dict)
    pairs_train["a2_rank"] = pairs_train["a2_id"].map(ranks_dict)
    pairs_train["label_pred"] = pairs_train.apply(
        lambda x: pairwise_predict(x), axis=1,
    )
    pairs_train_no_ties = pairs_train.dropna()

    results_accuracy = {
        "n": pairs_train.shape[0],
        "acc": accuracy_score(
            y_true=pairs_train_no_ties["label"],
            y_pred=pairs_train_no_ties["label_pred"],
        ),
        "acc_by_transition": [
            {
                "transition": transition,
                "n": pairs_train_no_ties[
                    pairs_train_no_ties["transition"] == transition
                ].shape[0],
                "acc": accuracy_score(
                    y_true=pairs_train_no_ties.loc[
                        pairs_train_no_ties["transition"] == transition, "label",
                    ],
                    y_pred=pairs_train_no_ties.loc[
                        pairs_train_no_ties["transition"] == transition, "label_pred",
                    ],
                ),
                "n_ties": pairs_train[
                    (
                        (pairs_train["a1_rank"] == pairs_train["a2_rank"])
                        & (pairs_train["transition"] == transition)
                    )
                ].shape[0],
            }
            for transition in ["rr", "rw", "wr", "ww"]
            if pairs_train_no_ties[
                pairs_train_no_ties["transition"] == transition
            ].shape[0]
            > 0
        ],
        "n_ties": pairs_train[pairs_train["a1_rank"] == pairs_train["a2_rank"]].shape[
            0
        ],
    }

    results = {
        "accuracies": results_accuracy,
        "rank_scores": ranks_dict,
    }

    if rank_score_type == "crowd_BT":
        results.update({"annotator_params": annotator_params})
    return results


def build_rankings_by_topic_over_time(topic, discipline, rank_score_type):
    """
    for all answers to a given question (a.k.a. topic), assign a "rank score";
    this method follows a time-series based validation, where the rankings are
    calculated at each time step, and tested for the pairs of subsequent student
    """

    pairs_df, df_topic = get_topic_data(topic=topic, discipline=discipline)

    accuracies, accuracies_train = [], []
    sorted_args = []
    annotator_params = []
    # rankings_by_batch = []
    rank_scores = []
    # rank_scores_by_batch = []

    steps = pairs_df["annotation_rank_by_time"].value_counts().shape[0]
    for counter, (r, df_r) in enumerate(pairs_df.groupby("annotation_rank_by_time")):
        if r % 20 == 0:
            print("\t\ttime step {}/{}".format(counter, steps,))

        pairs_train = pairs_df[pairs_df["annotation_rank_by_time"] < r].copy()
        pairs_test = pairs_df[pairs_df["annotation_rank_by_time"] == r].copy()

        df_train = df_topic[df_topic["a_rank_by_time"] < r].copy()

        students = pairs_train["annotator"].drop_duplicates().to_list()

        transition = df_topic.loc[df_topic["a_rank_by_time"] == r, "transition"].iat[0]
        annotator = df_topic.loc[df_topic["a_rank_by_time"] == r, "user_token"].iat[0]

        if pairs_train.shape[0] > 0 and len(students) > 10:

            results = get_ranking_model_fit(pairs_train, df_train, rank_score_type)

            results["accuracies"].update(
                {"transition": transition, "annotator": annotator, "r": r,}
            )
            for d in results["accuracies"]["acc_by_transition"]:
                d.update(
                    {"transition": transition, "annotator": annotator, "r": r,}
                )

            if rank_score_type == "crowd_BT":
                annotator_params.append(results["annotator_params"])

            # save for analysis
            accuracies_train.append(results["accuracies"])
            rank_scores.append(results["ranks_dict"])

            # test ability of rankings to predict winning argument in
            # held out pairs at current timestep
            pairs_test["a1_rank"] = pairs_test["a1_id"].map(results["ranks_dict"])
            pairs_test["a2_rank"] = pairs_test["a2_id"].map(results["ranks_dict"])

            # pairs with current student's argument must be dropped
            # as it is as yet unseen
            pairs_test_ = pairs_test[["a1_rank", "a2_rank", "label"]].dropna().copy()

            # for each pair in held out pairs at current timestep,
            # predict winner based on higher param
            if pairs_test_.shape[0] > 0:
                pairs_test_["label_pred"] = pairs_test_.apply(
                    lambda x: pairwise_predict(x), axis=1,
                )
                pairs_test_ = pairs_test_.dropna().copy()

                accuracies.append(
                    {
                        "r": r,
                        "n": pairs_test_.shape[0],
                        "acc": accuracy_score(
                            y_true=pairs_test_["label"],
                            y_pred=pairs_test_["label_pred"],
                        ),
                        "transition": transition,
                        "annotator": annotator,
                        "n_ties": pairs_test_[
                            pairs_test_["a1_rank"] == pairs_test_["a2_rank"]
                        ].shape[0],
                    }
                )
            # # make two batches of students, interleaved in time
            # student_batch1 = students[::2]
            # student_batch2 = [
            #     s for s in students if s not in student_batch1
            # ]
            #
            # # get rankings for each batch and save
            # batch_rankings, batch_rank_scores = {}, {}
            # for sb, student_batch in zip(
            #     ["batch1", "batch2"], [student_batch1, student_batch2]
            # ):
            #
            #     if rank_score_type=="baseline":
            #         df_train_batch = df_topic[df_topic["user_token"].isin(student_batch)]
            #         sorted_arg_ids, ranks_dict = get_rankings_baseline(
            #             df_train=df_train_batch
            #         )
            #     else:
            #         pairs_train_batch = pairs_train[
            #             pairs_train["annotator"].isin(student_batch)
            #         ]
            #         sorted_arg_ids, ranks_dict = get_rankings(
            #             pairs_train=pairs_train_batch
            #         )
            #
            #     batch_rankings[sb] = sorted_arg_ids
            #     batch_rank_scores[sb] = ranks_dict
            #
            # rankings_by_batch.append(batch_rankings)
            # rank_scores_by_batch.append(batch_rank_scores)

    results = {
        "accuracies": accuracies,
        "accuracies_train": accuracies_train,
        # "rankings_by_batch": rankings_by_batch,
        "rank_scores": rank_scores,
        # "rank_scores_by_batch" : rank_scores_by_batch,
    }
    if rank_score_type == "crowd_BT":
        results.update({"annotator_params": annotator_params})

    return results


def main(
    discipline: (
        "Discipline",
        "positional",
        None,
        str,
        ["Physics", "Biology", "Chemistry"],
    ),
    rank_score_type: (
        "Rank Score Type",
        "positional",
        None,
        str,
        ["BT", "elo", "crowd_BT", "winrate", "wc"],
    ),
    largest_first: ("Largest Files First", "flag", "l", bool,),
    time_series_validation_flag: ("Time Series Validation", "flag", "t", bool,),
):
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
    print("Discipline : {} - Rank Score Type: {}".format(discipline, rank_score_type))
    data_dir_discipline = get_data_dir(discipline)
    if time_series_validation_flag:
        results_dir_discipline = os.path.join(RESULTS_DIR, discipline, rank_score_type)
    else:
        results_dir_discipline = os.path.join(
            RESULTS_DIR, discipline, "model_fit", rank_score_type
        )
    # make results directories if they do not exist:
    if not os.path.exists(results_dir_discipline):
        Path(results_dir_discipline).mkdir(parents=True, exist_ok=True)
        os.mkdir(os.path.join(results_dir_discipline, "accuracies",))
        os.mkdir(os.path.join(results_dir_discipline, "accuracies_train",))
        # os.mkdir(os.path.join(results_dir_discipline, "rankings_by_batch",))
        os.mkdir(os.path.join(results_dir_discipline, "rank_scores",))
        # os.mkdir(os.path.join(results_dir_discipline, "rank_scores_by_batch",))
        os.mkdir(os.path.join(results_dir_discipline, "annotator_params",))
    # sort files by size to get smallest ones done first
    # https://stackoverflow.com/a/20253803
    all_files = (
        os.path.join(data_dir_discipline, filename)
        for basedir, dirs, files in os.walk(data_dir_discipline)
        for filename in files
    )
    all_files = sorted(all_files, key=os.path.getsize, reverse=largest_first)

    topics = [os.path.basename(fp)[:-4] for fp in all_files]

    topics_already_done = [
        t[:-5] for t in os.listdir(os.path.join(results_dir_discipline, "rank_scores",))
    ]

    topics = [t for t in topics if t not in topics_already_done]

    for i, topic in enumerate(topics):
        print("\t{}/{} {}".format(i, len(topics), topic))

        if time_series_validation_flag:
            # results will come be calculated for each time step
            results = build_rankings_by_topic_over_time(
                topic=topic, discipline=discipline, rank_score_type=rank_score_type
            )
            fp = os.path.join(
                results_dir_discipline, "accuracies_train", "{}.json".format(topic)
            )
            with open(fp, "w+") as f:
                json.dump(results["accuracies_train"], f, indent=2)
        else:
            # results only for all data on tis topic/question
            pairs_df, df_topic = get_topic_data(topic=topic, discipline=discipline)
            results = get_ranking_model_fit(
                pairs_train=pairs_df, df_train=df_topic, rank_score_type=rank_score_type
            )

        fp = os.path.join(results_dir_discipline, "accuracies", "{}.json".format(topic))
        with open(fp, "w+") as f:
            json.dump(results["accuracies"], f, indent=2)

        # fp = os.path.join(
        #     results_dir_discipline,"rankings_by_batch", "{}.json".format(topic),
        # )
        # with open(fp, "w+") as f:
        #     json.dump(results["rankings_by_batch"],f, indent=2)

        fp = os.path.join(
            results_dir_discipline, "rank_scores", "{}.json".format(topic),
        )
        with open(fp, "w+") as f:
            json.dump(results["rank_scores"], f, indent=2)

        # fp = os.path.join(
        #     results_dir_discipline,"rank_scores_by_batch", "{}.json".format(topic),
        # )
        # with open(fp, "w+") as f:
        #     json.dump(results["rank_scores_by_batch"],f, indent=2)

        if rank_score_type == "crowd_BT":
            fp = os.path.join(
                results_dir_discipline, "annotator_params", "{}.json".format(topic),
            )
            with open(fp, "w+") as f:
                json.dump(results["annotator_params"], f, indent=2)

    return


if __name__ == "__main__":

    plac.call(main)
