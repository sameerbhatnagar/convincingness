import os
import json
import pandas as pd

from argBT import (
    get_rankings_wc,
    get_rankings_winrate_no_pairs,
    get_rankings_winrate,
    get_rankings_BT,
    get_rankings_elo,
    get_rankings_crowdBT,
    get_rankings_reference,
)

from data_loaders import (
    DALITE_DISCIPLINES,
    ARG_MINING_DATASETS,
    BASE_DIR,
    MIN_ANSWERS,
    get_topic_data
)
from feature_extraction import append_features, append_wc_quartile_column


def load_data(discipline, output_dir_name,feature_types_included = ["surface", "lexical", "readability", "syntax", "semantic"],
topics_filtered=True,
):
    """
    load data and append features
    """
    if discipline in DALITE_DISCIPLINES:
        population="switchers" #"all"
    else:
        population="all"

    # load data and append features
    data_dir_discpline = os.path.join(
        BASE_DIR, "tmp", output_dir_name, discipline, population, "data"
    )
    output_dir = os.path.join(data_dir_discpline, os.pardir)
    if topics_filtered:
        fp = os.path.join(
            BASE_DIR, "tmp", output_dir_name, discipline, population, "topics.json"
        )
        with open(fp, "r") as f:
            topics = json.load(f)
    else:
        topics = os.listdir(data_dir_discpline)

    df = pd.DataFrame()
    print("\t a) loading data for each topic and appending features")
    for t, topic in enumerate(topics):

        topic = topic.replace(".csv", "")

        if t % (len(topics)//10) == 0:
            print(f"\t\t{t}/{len(topics)}")

        # `append_features` calls on `get_topic_data`, which will filter on
        # MIN_TIMES_SHOWN and MIN_WORD_COUNT
        df_topic_with_features = append_features(
            topic, discipline, feature_types_included, output_dir
        )
        df_topic_with_features["topic"]=topic

        # only work with topics with at least MIN_ANSWERS
        if (df_topic_with_features.shape[0]>MIN_ANSWERS) or (discipline in ARG_MINING_DATASETS):
            df_topic_with_features=append_wc_quartile_column(df_topic_with_features)

            # load pairs, which are needed for calculating
            # convincingenss scores
            pairs_df, _ = get_topic_data(
                topic=topic.replace(".csv", ""),
                discipline=discipline,
                output_dir=output_dir,
            )
            if discipline in DALITE_DISCIPLINES:
                df_topic_with_features["arg_id"] = "arg" + df_topic_with_features["id"].astype(
                    str
                )
            else:
                df_topic_with_features["arg_id"] =  df_topic_with_features["id"]
                df_topic_with_features["id"] = df_topic_with_features["id"].str.replace("arg","").map(int)

            # load targets
            _, args_dict = get_rankings_winrate(pairs_df)
            df_topic_with_features["y_winrate"] = df_topic_with_features["arg_id"].map(
                args_dict
            )

            _, args_dict = get_rankings_elo(pairs_df)
            df_topic_with_features["y_elo"] = df_topic_with_features["arg_id"].map(
                args_dict
            )

            _, args_dict = get_rankings_BT(pairs_df)
            df_topic_with_features["y_BT"] = df_topic_with_features["arg_id"].map(args_dict)

            if discipline in DALITE_DISCIPLINES:
                _, args_dict = get_rankings_winrate_no_pairs(df_topic_with_features)
                df_topic_with_features["y_winrate_nopairs"] = df_topic_with_features[
                    "arg_id"
                ].map(args_dict)

                _, args_dict, _ = get_rankings_crowdBT(pairs_df)
                df_topic_with_features["y_crowdBT"] = df_topic_with_features["arg_id"].map(
                    args_dict
                )
            else:
                args_dict = get_rankings_reference(topic,discipline,output_dir_name)
                df_topic_with_features["y_reference"] = df_topic_with_features["arg_id"].map(
                    args_dict
                )

            # combine targets and features
            df = pd.concat([df, df_topic_with_features])

        else:
            print(f"\t\t\tskip {t}:{topic} - {df_topic_with_features.shape[0]} answers")
    return df


def load_all_args_features_scores(output_dir_name):
    """
    for all explanations in disciplines/datasets,
    load features and target scores
    (~13 minutes to load)

    Arguments:
    ---------
        output_dir_name: where to look in `tmp` directory

    Returns:
    --------
        df_all_answers: pandas dataframe with all data+features+targets
    """

    df_all_answers=pd.DataFrame()
    feature_types_included = ["surface", "lexical", "readability", "syntax", "semantic"]

    for discipline in ARG_MINING_DATASETS+DALITE_DISCIPLINES:
        print(discipline)
        if discipline in DALITE_DISCIPLINES:
            population="switchers" # "all"
        else:
            population="all"

        output_dir = os.path.join(
            BASE_DIR, "tmp", output_dir_name, discipline, population
        )

        print("\t 1) Loading data")
        df=load_data(
            discipline=discipline,
            output_dir_name=output_dir_name,
            feature_types_included=feature_types_included,
            population=population
        )

        # use to get correlations between Y_reference and other y_targets
        df["discipline"]=discipline
        df_all_answers=pd.concat([df_all_answers,df])
    return df_all_answers
