import pandas as pd
import numpy as np

def make_summary_by_topic(df_topics_all,pairs_df_all):
    """
    given all answers and pairs, return means/stds per topic

    Returns:
    =======
        df_summary_by_topic

    """
    topics_N=df_topics_all.groupby(
        "discipline"
    )["topic"].value_counts().to_frame().rename(
        columns={"topic":"N"}
    )
    topics_N_pairs=pairs_df_all.groupby(
        "discipline"
    )["topic"].value_counts().to_frame().rename(
        columns={"topic":"N_pairs"}
    )

    df_summary_by_topic=pd.merge(
        topics_N,
        topics_N_pairs,
        left_index=True,
        right_index=True
    ).sort_values("N").reset_index()

    topics_wc=df_topics_all.groupby("topic")["surface_n_words"].mean().to_frame().rename(
        columns={"surface_n_words":"surface_n_words_mean"}
    )
    df_summary_by_topic=pd.merge(
        df_summary_by_topic,
        topics_wc,
        on="topic"
    )
    topics_std=df_topics_all.groupby("topic")["surface_n_words"].std().to_frame().rename(
        columns={"surface_n_words":"surface_n_words_std"}
    )

    df_summary_by_topic=pd.merge(
        df_summary_by_topic,
        topics_std,
        on="topic"
    )

    # topics_wc_diff=pairs_df_all.groupby("topic")["wc_diff"].mean().to_frame().rename(
    #     columns={"wc_diff":"wc_diff_mean"}
    # )
    # df_summary_by_topic=pd.merge(
    #     df_summary_by_topic,
    #     topics_wc_diff,
    #     on="topic"
    # )
    # topics_wc_diff_std=pairs_df_all.groupby("topic")["wc_diff"].std().to_frame().rename(
    #     columns={"wc_diff":"wc_diff_std"}
    # )
    # df_summary_by_topic=pd.merge(
    #     df_summary_by_topic,
    #     topics_wc_diff_std,
    #     on="topic"
    # )
    return df_summary_by_topic


def get_mean_times_shown(pairs_df):
    """
    return how many times arguments appeared in pairs
    """
    return pd.concat([pairs_df["a1_id"], pairs_df["a2_id"]]).value_counts().describe()["mean"].round()

def means_by_topic(x):
    """
    function to be used on
     df_summary_by_topic.groupby("discipline").apply()

    calculate base statistics
    """
    d={}
    d["topics"]=x.shape[0]
    d["args"] = x["N"].sum()
    d["pairs"] = x["N_pairs"].sum()
    d["args/topic"]=f"{x['N'].mean():.0f} ({x['N'].std():.0f})"
    d["pairs/topic"]=f"{x['N_pairs'].mean():.0f} ({x['N_pairs'].std():.0f})"
    d["pairs/arg"]=f"{np.average(x['pairs/arg'],weights=x['N']):.0f} ({x['pairs/arg'].std():.0f})"
    d["wc"] =  f"{np.average(x['surface_n_words_mean'],weights=x['N']):.0f} ({x['surface_n_words_mean'].std():.0f})"
    # d["wc_diff"] =  f"{np.average(x['wc_diff_mean'],weights=x['N']):.0f} ({x['wc_diff_mean'].std():.0f})"
    return pd.Series(d,index=[
        "topics",
        "args",
        "pairs",
        "args/topic",
        "pairs/topic",
        "pairs/arg",
        "wc",
        # "wc_diff (+/-)",
    ])
