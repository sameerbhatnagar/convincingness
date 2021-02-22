import os
import json
import plac
import math
from pathlib import Path
import numpy as np
import pandas as pd

import spacy

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

MIN_VOTES, MIN_SHOWN = 1, 8


def prior_factory(key):
    return lambda: PRIORS[key]


only_one_arg_pair = {}
model = "argBT"

MAX_ITERATIONS = 1500
MIN_WORD_COUNT_DIFF = 5
TRANSITIONS = {
    "Ethics": ["same_ans", "switch_ans"],
    "Physics": ["rr", "wr", "rw", "ww"],
    "Chemistry": ["rr", "wr", "rw", "ww"],
    "same_teacher_two_groups":["rr", "wr", "rw", "ww"],
}

DALITE_DISCIPLINES = ["Physics","Chemistry","Ethics"]

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
    nlp = spacy.load("en_core_web_md")
    
    rationales = df_train[["rationale", "id"]].values
    ranks_dict = {
        "arg{}".format(arg_id): len(
            [token for token in doc if token.pos_ not in DROPPED_POS]
        )
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }
    sorted_arg_ids = [
        k for k, v in sorted(ranks_dict.items(), key=lambda x: x[1], reverse=True)
    ]

    return sorted_arg_ids, ranks_dict


def get_rankings_reference(topic,discipline,output_dir_name):
    """
    Used to get pre-computed rankings for reference datasets
    (UKP, IBM_ArgQ, IBM_Evi)
    Returns:
    -------
        - args_dict
    """
    population="all"

    output_dir = os.path.join(
        data_loaders.BASE_DIR, "tmp", output_dir_name, discipline, population
    )
    if discipline in ["UKP","IBM_ArgQ"]:
        if discipline =="UKP":
            fp=os.path.join(data_loaders.DATASETS[discipline]["rank_data_dir"],f"{topic}.csv")
        elif discipline=="IBM_ArgQ":
            fp=os.path.join(data_loaders.DATASETS[discipline]["rank_data_dir"],f"{topic}.tsv")
        df_topic_ranks=pd.read_csv(fp,sep="\t")

        _,df_topic=data_loaders.get_topic_data(topic=topic,discipline=discipline,output_dir=output_dir)

        # ids from reference datasets do not always match between rank-score files and arg-pairs files,
        # so join on argument text itself (first 40 caharcters)
        max_len=40
        df_topic_ranks["rationale_short"]=df_topic_ranks["argument"].str.lower().str.replace(" ","").str[:max_len]
        df_topic["rationale_short"]=df_topic["rationale"].str.lower().str.replace(" ","").str[:max_len]
        args_dict={
            k:v
            for k,v in pd.merge(
                    df_topic_ranks[["rank","rationale_short"]],
                    df_topic[["id","rationale_short"]],
                    on="rationale_short"
            )[["id","rank"]].values
        }
    elif discipline=="IBM_Evi":
        pairs_df,_=data_loaders.get_topic_data(topic=topic,discipline=discipline,output_dir=output_dir)
        args_dict={
                k:v for
                k,v in
                pd.concat([
                    pairs_df[["a1_id","a1_rank"]].rename(columns={"a1_id":"id","a1_rank":"rank"}),
                    pairs_df[["a2_id","a2_rank"]].rename(columns={"a2_id":"id","a2_rank":"rank"})
                ]).drop_duplicates(["id"]).values
        }
    return args_dict

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
        - annotator_params
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


def get_rankings_crowdBT_filtered(pairs_train):
    """
    recalculate argument strengths after filtering out least reliable annotators (>50th percentile)
    """
    sorted_arg_ids, ranks_dict, annotator_params = get_rankings_crowdBT(pairs_train)
    thresh = np.percentile(list(annotator_params.values()), 50)

    trustworthy = [a for a, v in annotator_params.items() if v > thresh]
    pairs_train_trustworthy = pairs_train[
        (pairs_train["annotator"].isin(trustworthy))
    ].copy()

    sorted_arg_ids, ranks_dict, annotator_params = get_rankings_crowdBT(
        pairs_train_trustworthy
    )

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
    Tuple:
        - sorted_arg_ids : array or arg_ids sorted by rank_score
        - ranks_dict: dict where keys are arg_id, and values are BT rank scores
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


def get_rankings_winrate(pairs_df):
    """
    given pairs, simply count times won/times shown for each arg
    "Laplace" transform, where assume objects never shown would have
    a MIN_VOTES/MIN_SHOWN rank score
    """
    times_shown = pd.concat([pairs_df["a1_id"], pairs_df["a2_id"]]).value_counts()
    times_chosen = (
        pairs_df.apply(
            lambda x: x["a1_id"] if x["label"] == "a1" else x["a2_id"], axis=1
        )
    ).value_counts()
    ranks_dict = ((times_chosen) / (times_shown)).to_dict()

    # need to add entries for rationales never chosen
    never_chosen = [
        r for r in times_shown.index.to_list() if r not in times_chosen.index.to_list()
    ]
    ranks_dict.update({r: 0 for r in never_chosen})

    sorted_arg_ids = [
        k for k, v in sorted(ranks_dict.items(), key=lambda x: x[1], reverse=True)
    ]
    return sorted_arg_ids, ranks_dict


def get_rankings_winrate_no_pairs(df_train):
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
        "arg{}".format(k): (v) / (times_shown_counter[k]+1)
        for k, v in votes_count.items()
    }
    # need to add entries for rationales never chosen
    never_chosen = [r for r in df_train["id"].to_list() if r not in votes_count]
    ranks_dict.update({"arg{}".format(r): 0 for r in never_chosen})
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


def get_ranking_model_fit(pairs_train, df_train, rank_score_type, discipline):
    """
    calculate rankings based on specified type, and run pairwise prediction
    on training pairs
    """

    if rank_score_type == "winrate":
        # ranking = times_chosen/times_shown
        sorted_arg_ids, ranks_dict = get_rankings_winrate(pairs_df=pairs_train)

    if rank_score_type == "winrate_no_pairs":
        # ranking = times_chosen/times_shown
        sorted_arg_ids, ranks_dict = get_rankings_winrate_no_pairs(df_train=df_train)

    elif rank_score_type == "wc":
        # ranking = num tokens
        sorted_arg_ids, ranks_dict = get_rankings_wc(df_train=df_train)

    elif rank_score_type == "crowd_BT":
        # ranking + annotator params based on crowd_BT
        sorted_arg_ids, ranks_dict, annotator_params = get_rankings_crowdBT(
            pairs_train=pairs_train
        )
    elif rank_score_type == "crowdBT_filtered":
        # rankings after filtering out bottom 10 percentile annotators
        sorted_arg_ids, ranks_dict, annotator_params = get_rankings_crowdBT_filtered(
            pairs_train=pairs_train
        )

    elif rank_score_type == "elo":
        # ranking based on ELO match-ups, mean = 1500
        sorted_arg_ids, ranks_dict = get_rankings_elo(pairs_df=pairs_train)

    elif rank_score_type == "BT":
        # rankings from conventional Bradley Terry
        sorted_arg_ids, ranks_dict = get_rankings_BT(pairs_train=pairs_train)

    elif rank_score_type == "reference":
        # FIX ME
        output_dir_name="exp2"
        topic = pairs_train["topic"].value_counts().index.to_list()[0]
        ranks_dict = get_rankings_reference(topic,discipline,output_dir_name)

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
        "n_ties": pairs_train[pairs_train["a1_rank"] == pairs_train["a2_rank"]].shape[0]
        }
    if discipline in DALITE_DISCIPLINES:
        results_accuracy.update({
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
            if pairs_train_no_ties[
                pairs_train_no_ties["transition"] == transition
            ].shape[0]
            > 0
            else {
                "transition": transition,
                "n": 0,
                "acc": None,
                "n_ties": pairs_train[
                    (
                        (pairs_train["a1_rank"] == pairs_train["a2_rank"])
                        & (pairs_train["transition"] == transition)
                    )
                ].shape[0],
            }
            for transition in TRANSITIONS[discipline]
        ],
    })

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

    pairs_df, df_topic = data_loaders.get_topic_data(topic=topic, discipline=discipline,output_dir=output_dir)

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

            results = get_ranking_model_fit(
                pairs_train, df_train, rank_score_type, discipline
            )

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

    results = {
        "accuracies": accuracies,
        "accuracies_train": accuracies_train,
        "rank_scores": rank_scores,
    }
    if rank_score_type == "crowd_BT":
        results.update({"annotator_params": annotator_params})

    return results


def get_model_fit_by_batch(df_topic, pairs_df, rank_score_type,discipline):
    """
    find rank scores and accuracies using two-fold Validation
    with interleaved batches of students, by transition type

    Returns
    =======
    Tuple of dicts, where keys are transition type
        batch_accuracies: dict
        batch_rank_scores: dict
    """
    batch_rank_scores, batch_accuracies = {}, {}

    for transition in TRANSITIONS[discipline]:
        df_transition = df_topic[df_topic["transition"] == transition].copy()
        students = df_transition.sort_values("id")["user_token"].dropna().to_list()
        # if len(students) >= 10:
        # make two batches of students, interleaved in time
        student_batch1 = students[::2]
        student_batch2 = [s for s in students if s not in student_batch1]
        batch_rank_scores[transition], batch_accuracies[transition] = {}, {}
        # print("\t{}".format(transition))
        # get rankings for each batch and save
        for sb, student_batch in zip(
            ["batch1", "batch2"], [student_batch1, student_batch2]
        ):
            df_train_batch = df_transition[
                df_transition["user_token"].isin(student_batch)
            ].copy()
            pairs_train_batch = pairs_df[
                pairs_df["annotator"].isin(student_batch)
            ].copy()
            if pairs_train_batch.shape[0] > 0:
                rb = get_ranking_model_fit(
                    pairs_train=pairs_train_batch,
                    df_train=df_train_batch,
                    rank_score_type=rank_score_type,
                    discipline=discipline,
                )
                # just keep the rank scores for each batch
                batch_rank_scores[transition][sb] = rb["rank_scores"]

                # test ability of rankings to predict winning argument in
                # other batch
                if sb == "batch1":
                    other_batch = student_batch2
                else:
                    other_batch = student_batch1
                pairs_test = pairs_df[
                    pairs_df["annotator"].isin(other_batch)
                ].copy()
                pairs_test["a1_rank"] = pairs_test["a1_id"].map(rb["rank_scores"])
                pairs_test["a2_rank"] = pairs_test["a2_id"].map(rb["rank_scores"])

                # pairs with current student's argument must be dropped
                # as it is as yet unseen
                pairs_test_ = (
                    pairs_test[["a1_rank", "a2_rank", "label"]].dropna().copy()
                )

                if pairs_test_.shape[0] > 0:
                    # print(
                    #     "\t\t {}:{} pairs, {} students".format(
                    #         sb, pairs_test_.shape[0], len(student_batch)
                    #     )
                    # )
                    # for each pair in held out pairs,
                    # predict winner based on higher param
                    pairs_test_["label_pred"] = pairs_test_.apply(
                        lambda x: pairwise_predict(x), axis=1,
                    )
                    pairs_test_ = pairs_test_.dropna().copy()

                    batch_accuracies[transition][sb] = {
                        "n": pairs_test_.shape[0],
                        "acc": accuracy_score(
                            y_true=pairs_test_["label"],
                            y_pred=pairs_test_["label_pred"],
                        ),
                        "transition": transition,
                        "n_ties": pairs_test_[
                            pairs_test_["a1_rank"] == pairs_test_["a2_rank"]
                        ].shape[0],
                    }
    return batch_accuracies, batch_rank_scores


def main(
    discipline: (
        "Discipline",
        "positional",
        None,
        str,
        ["Physics", "Chemistry", "Ethics", "same_teacher_two_groups", "UKP", "IBM_ArgQ"],
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
    output_dir_name: (
        "Directory name for results",
        "positional",
        None,
        str,
    ),
    largest_first: ("Largest Files First", "flag", "largest-first", bool,),
    time_series_validation_flag: ("Time Series Validation", "flag", "time-series", bool,),
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
    # if filter_switchers:
    #     output_dir = os.path.join(data_loaders.BASE_DIR, "tmp", output_dir_name, discipline,"switchers")
    # else:
    output_dir = os.path.join(data_loaders.BASE_DIR, "tmp", output_dir_name, discipline,"all")

    data_dir_discipline = os.path.join(output_dir,"data")
    if time_series_validation_flag:
        results_dir_discipline = os.path.join(output_dir,"time_series", rank_score_type)
    else:
        results_dir_discipline = os.path.join(
            output_dir, "model_fit", rank_score_type
        )
    # make results directories if they do not exist:
    Path(results_dir_discipline).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir_discipline, "accuracies")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(results_dir_discipline, "accuracies_train",)).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(results_dir_discipline, "rank_scores",)).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(results_dir_discipline, "rank_scores_by_batch")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(results_dir_discipline, "accuracies_by_batch")).mkdir(
        parents=True, exist_ok=True
    )
    if time_series_validation_flag:
        Path(os.path.join(results_dir_discipline, "accuracies_by_time")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(results_dir_discipline, "rankings_by_time")).mkdir(
            parents=True, exist_ok=True
        )
    Path(os.path.join(results_dir_discipline, "annotator_params")).mkdir(
        parents=True, exist_ok=True
    )
    # sort files by size to get smallest ones done first
    # https://stackoverflow.com/a/20253803
    all_files = (
        os.path.join(data_dir_discipline, filename)
        for basedir, dirs, files in os.walk(data_dir_discipline)
        for filename in files
    )
    all_files = sorted(all_files, key=os.path.getsize, reverse=largest_first)

    topics = [os.path.basename(fp)[:-4] for fp in all_files]

    if time_series_validation_flag:
        topics_already_done = [
            t[:-5] for t in os.listdir(os.path.join(results_dir_discipline, "rankings_by_time",))
        ]
    else:
        topics_already_done = [
            t[:-5] for t in os.listdir(os.path.join(results_dir_discipline, "rank_scores",))
        ]

    topics = [t for t in topics if t not in topics_already_done]

    for i, topic in enumerate(topics):
        print("\t{}/{} {}".format(i, len(topics), topic))
        # results only for all data on this topic/question

        pairs_df, df_topic = data_loaders.get_topic_data(topic=topic, discipline=discipline,output_dir=output_dir)

        # pairs_df=pairs_df.sort_values("annotation_rank_by_time").groupby("#id").head(MAX_PAIR_OCCURENCES).copy()

        print("\t {} pairs, {} students".format(pairs_df.shape[0], df_topic.shape[0]))

        if time_series_validation_flag:

            accuracies_all, rank_scores_all = [], []

            steps = pairs_df["annotation_rank_by_time"].value_counts().shape[0]
            for counter, (r, df_r) in enumerate(
                pairs_df.groupby("annotation_rank_by_time")
            ):
                if r % 20 == 0:
                    print("\t\ttime step {}/{}".format(counter, steps,))

                if r >= 40:

                    df_timestep = df_topic[df_topic["a_rank_by_time"] <= r]
                    pairs_df_timestep = pairs_df[
                        pairs_df["annotation_rank_by_time"] < r
                    ].copy()

                    results = get_ranking_model_fit(
                        df_train=df_timestep,
                        pairs_train=pairs_df_timestep,
                        rank_score_type=rank_score_type,
                        discipline=discipline,
                    )

                    rank_scores_all.append(results["rank_scores"])
                    accuracies_all.append(results["accuracies"])

            fp = os.path.join(
                results_dir_discipline, "accuracies_by_time", "{}.json".format(topic)
            )
            with open(fp, "w+") as f:
                json.dump(accuracies_all, f, indent=2)

            fp = os.path.join(
                results_dir_discipline, "rankings_by_time", "{}.json".format(topic)
            )
            with open(fp, "w+") as f:
                json.dump(rank_scores_all, f, indent=2)

        else:
            results = get_ranking_model_fit(
                pairs_train=pairs_df,
                df_train=df_topic,
                rank_score_type=rank_score_type,
                discipline=discipline,
            )
            if discipline in DALITE_DISCIPLINES:
                batch_accuracies, batch_rank_scores = get_model_fit_by_batch(
                    df_topic=df_topic,
                    pairs_df=pairs_df,
                    rank_score_type=rank_score_type,
                    discipline=discipline,
                )
                # rank_scores_by_batch
                fp = os.path.join(
                    results_dir_discipline, "rank_scores_by_batch", "{}.json".format(topic),
                )
                with open(fp, "w+") as f:
                    json.dump(batch_rank_scores, f, indent=2)

                # accuracies by batch
                fp = os.path.join(
                    results_dir_discipline, "accuracies_by_batch", "{}.json".format(topic),
                )
                with open(fp, "w+") as f:
                    json.dump(batch_accuracies, f, indent=2)

            # pairwise classification results
            fp = os.path.join(
                results_dir_discipline, "accuracies", "{}.json".format(topic)
            )
            with open(fp, "w+") as f:
                json.dump(results["accuracies"], f, indent=2)

            # rank scores
            fp = os.path.join(
                results_dir_discipline, "rank_scores", "{}.json".format(topic),
            )
            with open(fp, "w+") as f:
                json.dump(results["rank_scores"], f, indent=2)

            # crowd_BT give annotator params as well
            if rank_score_type == "crowd_BT":
                fp = os.path.join(
                    results_dir_discipline, "annotator_params", "{}.json".format(topic),
                )
                with open(fp, "w+") as f:
                    json.dump(results.get("annotator_params"), f, indent=2)

    return


if __name__ == "__main__":

    plac.call(main)
