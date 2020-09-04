import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from feature_extraction import MIN_WORD_COUNT
from feature_extraction import PRE_CALCULATED_FEATURES
from utils import get_vocab

def shown_feature_counts(shown_ids_str, df, feature):
    avg_feature_value = df[feature].mean()
    try:
        shown_ids = shown_ids_str.strip("[]").replace(" ", "").split(",")
    except AttributeError:
        shown_ids = []
    shown_wc = []
    for k in shown_ids:
        if k != "":
            try:
                shown_wc.append(
                    df.loc[df["id"] == int(k), feature].iat[0]
                    if not (df.loc[df["id"] == int(k), feature].isna().iat[0])
                    else avg_feature_value
                )
            except IndexError:  # data got filtered out
                shown_wc.append(avg_feature_value)
    return shown_wc


def append_features_shown_LSA(df_q):
    """
    Arguments:
    =========
        df_q: all answers for a particular question, with "rationale" and shown
                "rationales" columns. Ordered by "a_rank_by_time" (i.e.
                when the answer was put in DB)
    Returns:
    =======
        df_q : add columns for <mean, max> LSA distances to
                - shown "rationales", all other rationales
    """
    # make LSA space with all rationales
    corpus = df_q["rationale"]
    vocab = get_vocab(corpus)
    n_components = 100 if len(vocab) > 100 else len(vocab) - 1
    lsa = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("svd", TruncatedSVD(n_components=n_components)),
            ("norm", Normalizer()),
        ]
    )

    lsa.fit(corpus)

    # iterate through each student answer
    for rank in df_q["a_rank_by_time"].values[5:]:

        # look up vector for each shown answer and append cos sim
        if len(df_q[df_q["a_rank_by_time"] == rank].rationales) > 0:
            shown_ids = [
                int(s)
                for s in df_q[df_q["a_rank_by_time"] == rank]
                .rationales.iat[0]
                .strip("[]")
                .replace(" ", "")
                .split(",")
            ]

            try:
                df_q.loc[
                    df_q["a_rank_by_time"] == rank, "mean_cosine_sim_to_shown"
                ] = cosine_similarity(
                    lsa.transform(df_q[df_q["a_rank_by_time"] == rank].rationale),
                    lsa.transform(df_q.loc[shown_ids, "rationale"]),
                ).mean()
            except KeyError:
                pass  #

        # add mean cosine similarity to all rationales
        df_q.loc[
            df_q["a_rank_by_time"] == rank, "mean_cosine_sim_to_others"
        ] = cosine_similarity(
            lsa.transform(df_q[df_q["a_rank_by_time"] == rank].rationale),
            lsa.transform(df_q["rationale"]),
        ).mean()


def append_features_shown(df, kwargs):
    """
    for each feature in  pre_calculated_features,
    for each answer, aggregate feature values for those rationales that were shown
    Same for LSA vectors
    """

    LSA_features = kwargs.get("LSA_features")
    pre_calculated_features = expand_feature_types_included(
        feature_types_included=kwargs.get("feature_types_included")
    )

    for feature in pre_calculated_features:
        # e.g. shown_convincingness_baseline will be array of values
        feature_name = "shown_{}".format(feature)
        # print("\t appending column for {}".format(feature_name))
        df[feature_name] = df["rationales"].apply(
            shown_feature_counts, args=(df, feature,)
        )

        # for the answers where we have "shown rationale data"
        # take the <max,min,mean> of shown_<feature> array (from previous step)
        # as feature
        df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), feature_name + "_mean"
        ] = df.loc[df[feature_name].apply(lambda x: len(x) > 0), feature_name].apply(
            lambda x: np.array(x).mean()
        )

        df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), feature_name + "_max"
        ] = df.loc[df[feature_name].apply(lambda x: len(x) > 0), feature_name].apply(
            lambda x: np.array(x).max()
        )

        df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), feature_name + "_min"
        ] = df.loc[df[feature_name].apply(lambda x: len(x) > 0), feature_name].apply(
            lambda x: np.array(x).min()
        )

    if "rationale_word_count" in pre_calculated_features:
        # take the number of shown rationales with word_count<MIN_WORD_COUNT
        # as feature
        feature = "rationale_word_count"
        feature_name="shown_{}".format(feature)
        df.loc[df[feature_name].apply(lambda x: len(x) > 0), "n_shown_short"] = df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), feature_name
        ].apply(lambda x: len([s for s in x if s < MIN_WORD_COUNT]))

        # take the number of shown rationales with word_count less than 0.5 times
        # their own rationale_word_count as feature
        df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), "n_shown_shorter_than_own"
        ] = df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), [feature, feature_name]
        ].apply(
            lambda x: len([i for i in x[feature_name] if i < 0.5 * x[feature]]), axis=1,
        )

        # take the number of shown rationales with word_count greater than 1.5 times
        # their own rationale_word_count as feature
        df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), "n_shown_longer_than_own"
        ] = df.loc[
            df[feature_name].apply(lambda x: len(x) > 0), [feature, feature_name]
        ].apply(
            lambda x: len([i for i in x[feature_name] if i > 1.5 * x[feature]]), axis=1,
        )

    df_q = df.sort_values("a_rank_by_time")

    # features based on cosine similarity of LSA vectors
    # (TfIdf + SVD + Norm)
    if LSA_features:
        df_q["a_rank_by_time"] = list(range(df_q.shape[0]))
        df_q = append_features_shown_LSA(df_q)

    return df_q

def expand_feature_types_included(feature_types_included):
    return [
        x
        for x in [
            PRE_CALCULATED_FEATURES[feature_type]
            for feature_type in feature_types_included
        ]
        for x in x
    ]


def get_feature_names(feature_types_included,discipline):
    """

    """

    pre_calculated_features = expand_feature_types_included(feature_types_included)

    feature_columns_numeric = (
        pre_calculated_features
        + ["shown_{}_mean".format(feature) for feature in pre_calculated_features]
        + ["shown_{}_max".format(feature) for feature in pre_calculated_features]
        + ["shown_{}_min".format(feature) for feature in pre_calculated_features]
    )

    if "surface" in feature_types_included:
        feature_columns_numeric.extend([
            # "a_rank_by_time",
            "n_shown_short",
            "n_shown_shorter_than_own",
            "n_shown_longer_than_own",
            # "q_diff1",
            # "student_strength1",
        ])

    if discipline != "Ethics":
        feature_columns_categorical = [
            "first_correct",
        ]
    else:
        feature_columns_categorical = []


    return feature_columns_numeric,feature_columns_categorical
