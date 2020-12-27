import pandas as pd
import os
from collections import Counter
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_DIR = os.path.join(BASE_DIR, "data")

DALITE_DISCIPLINES=["Ethics","Physics","Chemistry"]
ARG_MINING_DATASETS=["UKP","IBM_ArgQ","IBM_Evi"]

MIN_TIMES_SHOWN = 5
MIN_ANSWERS=100
MIN_PAIRS = 200
MIN_WORD_COUNT = 10

DATASETS = {}

DATASETS["IBM_ArgQ"] = {}
DATASETS["IBM_ArgQ"]["data_dir"] = os.path.join(BASE_DATA_DIR, "IBM-ArgQ-9.1kPairs")
DATASETS["IBM_ArgQ"]["rank_data_dir"] = os.path.join(BASE_DATA_DIR, "IBM-ArgQ-6.3kArgs")

DATASETS["IBM_ArgQ"]["files"] = [
    ("vaccination_PRO", "Flu-vaccination-should-be-mandatory-(PRO).tsv"),
    ("vaccination_CON", "Flu-vaccination-should-not-be-mandatory-(CON).tsv"),
    ("gambling_PRO", "Gambling-should-be-banned-(PRO).tsv"),
    ("gambling_CON", "Gambling-should-not-be-banned-(CON).tsv"),
    ("shopping_CON", "Online-shopping-brings-more-good-than-harm-(CON).tsv"),
    ("shopping_PRO", "Online-shopping-brings-more-harm-than-good-(PRO).tsv"),
    ("social_PRO", "Social-media-brings-MORE-GOOD-than-harm-(PRO).tsv"),
    ("social_CON", "Social-media-brings-MORE-HARM-than-good-(CON).tsv"),
    ("cryptocurrency_CON", "We-should-abandon-cryptocurrency-(CON).tsv"),
    ("vegetarianism_CON", "We-should-abandon-vegetarianism-(CON).tsv"),
    ("cryptocurrency_PRO", "We-should-adopt-cryptocurrency-(PRO).tsv"),
    ("vegetarianism_PRO", "We-should-adopt-vegetarianism-(PRO).tsv"),
    (
        "gaming_CON",
        "We-should-allow-the-sale-of-violent-video-games-to-minors-(CON).tsv",
    ),
    ("doping_CON", "We-should-ban-doping-in-sport-(CON).tsv"),
    ("fossil_PRO", "We-should-ban-fossil-fuels-(PRO).tsv"),
    ("gaming_PRO", "We-should-ban-the-sale-of-violent-video-games-to-minors-(PRO).tsv"),
    ("privacy_CON", "We-should-discourage-information-privacy-laws-(CON).tsv"),
    ("doping_PRO", "We-should-legalize-doping-in-sport-(PRO).tsv"),
    ("autonomous_PRO", "We-should-limit-autonomous-cars-(PRO).tsv"),
    ("fossil_CON", "We-should-not-ban-fossil-fuels-(CON).tsv"),
    ("autonomous_CON", "We-should-promote-autonomous-cars-(CON).tsv"),
    ("privacy_PRO", "We-should-support-information-privacy-laws-(PRO).tsv"),
]


DATASETS["UKP"] = {}
DATASETS["UKP"]["data_dir"] = os.path.join(BASE_DATA_DIR, "UKPConvArg1Strict-CSV")
DATASETS["UKP"]["rank_data_dir"] = os.path.join(
    BASE_DATA_DIR, "UKPConvArg1-Ranking-CSV"
)

DATASETS["UKP"]["files"] = [
    ("plastic_CON", "ban-plastic-water-bottles_no-bad-for-the-economy.csv"),
    ("plastic_PRO", "ban-plastic-water-bottles_yes-emergencies-only.csv"),
    ("christianity_PRO", "christianity-or-atheism-_atheism.csv"),
    ("christianity_CON", "christianity-or-atheism-_christianity.csv"),
    ("evolution_CON", "evolution-vs-creation_creation.csv"),
    ("evolution_PRO", "evolution-vs-creation_evolution.csv"),
    (
        "firefox_PRO",
        "firefox-vs-internet-explorer_it-has-a-cute-logo-oh-and-extensions-err-add-ons.csv",
    ),
    (
        "firefox_CON",
        "firefox-vs-internet-explorer_there-s-more-browsers-than-the-ie-firefox-is-an-animal.csv",
    ),
    (
        "gaymarriage_PRO",
        "gay-marriage-right-or-wrong_allowing-gay-marriage-is-right.csv",
    ),
    (
        "gaymarriage_CON",
        "gay-marriage-right-or-wrong_allowing-gay-marriage-is-wrong.csv",
    ),
    (
        "spanking_PRO",
        "human-growth-and-development-should-parents-use-spanking-as-an-option-to-discipline-_no.csv",
    ),
    (
        "spanking_CON",
        "human-growth-and-development-should-parents-use-spanking-as-an-option-to-discipline-_yes.csv",
    ),
    (
        "spouse_CON",
        "if-your-spouse-committed-murder-and-he-or-she-confided-in-you-would-you-turn-them-in-_no.csv",
    ),
    (
        "spouse_PRO",
        "if-your-spouse-committed-murder-and-he-or-she-confided-in-you-would-you-turn-them-in-_yes.csv",
    ),
    ("india_PRO", "india-has-the-potential-to-lead-the-world-_no-against.csv"),
    ("india_CON", "india-has-the-potential-to-lead-the-world-_yes-for.csv"),
    (
        "father_CON",
        "is-it-better-to-have-a-lousy-father-or-to-be-fatherless-_fatherless.csv",
    ),
    (
        "father_PRO",
        "is-it-better-to-have-a-lousy-father-or-to-be-fatherless-_lousy-father.csv",
    ),
    ("porn_CON", "is-porn-wrong-_no-is-is-not.csv"),
    ("porn_PRO", "is-porn-wrong-_yes-porn-is-wrong.csv"),
    ("uniform_PRO", "is-the-school-uniform-a-good-or-bad-idea-_bad.csv"),
    ("uniform_CON", "is-the-school-uniform-a-good-or-bad-idea-_good.csv"),
    (
        "common_PRO",
        "personal-pursuit-or-advancing-the-common-good-_advancing-the-commond-good.csv",
    ),
    (
        "common_CON",
        "personal-pursuit-or-advancing-the-common-good-_personal-pursuit.csv",
    ),
    ("abortion_PRO", "pro-choice-vs-pro-life_pro-choice.csv"),
    ("abortion_CON", "pro-choice-vs-pro-life_pro-life.csv"),
    ("physed_CON", "should-physical-education-be-mandatory-in-schools-_no-.csv"),
    ("physed_PRO", "should-physical-education-be-mandatory-in-schools-_yes-.csv"),
    ("tv_PRO", "tv-is-better-than-books_books.csv"),
    ("tv_CON", "tv-is-better-than-books_tv.csv"),
    (
        "farquhar_PRO",
        "william-farquhar-ought-to-be-honoured-as-the-rightful-founder-of-singapore_no-it-is-raffles-.csv",
    ),
    (
        "farquhar_CON",
        "william-farquhar-ought-to-be-honoured-as-the-rightful-founder-of-singapore_yes-of-course-.csv",
    ),
]


def get_cross_topic_validation_df(df_all):

    # df_all = df_all.rename(columns={"question": "topic"})

    N_folds = len(df_all["topic"].value_counts().index.to_list())
    train_dataframes = [pd.DataFrame() for _ in itertools.repeat(None, N_folds)]
    test_dataframes = [pd.DataFrame() for _ in itertools.repeat(None, N_folds)]

    for i, (topic, df_topic) in enumerate(df_all.groupby("topic")):
        train_dataframes[i] = pd.concat(
            [train_dataframes[i], df_all[df_all["topic"] != topic]]
        )
        test_dataframes[i] = pd.concat([test_dataframes[i], df_topic])
    return train_dataframes, test_dataframes


def get_arg_ranks(pairs_df, rank_data_dir, fname):
    # load ground truth rankings
    args_df = pd.read_csv(os.path.join(BASE_DATA_DIR, rank_data_dir, fname), sep="\t")
    args_df["_argument"] = args_df["argument"].str.lower().str.strip()

    # append pair_id column to table of ground truth rankings
    args_from_pairs_df = pd.concat(
        [
            pairs_df[["a1", "a1_id"]].rename(
                columns={"a1": "argument", "a1_id": "id_pair"}
            ),
            pairs_df[["a2", "a2_id"]].rename(
                columns={"a2": "argument", "a2_id": "id_pair"}
            ),
        ]
    ).drop_duplicates("id_pair")

    args_from_pairs_df["_argument"] = (
        args_from_pairs_df["argument"].str.lower().str.strip()
    )

    if "IBM" in rank_data_dir:
        args_df = pd.merge(args_df, args_from_pairs_df, on="_argument",)
        arg_rank_id_dict = (
            args_df[["id_pair", "rank"]].set_index("id_pair").to_dict()["rank"]
        )

    elif "UKP" in rank_data_dir:
        args_df["#id"] = args_df["#id"].astype("str")

        args_df = pd.merge(
            args_df, args_from_pairs_df, left_on="#id", right_on="id_pair"
        )

        arg_rank_id_dict = args_df[["#id", "rank"]].set_index("#id").to_dict()["rank"]

    pairs_df["a1_rank"] = pairs_df["a1_id"].map(arg_rank_id_dict)
    pairs_df["a2_rank"] = pairs_df["a2_id"].map(arg_rank_id_dict)

    return pairs_df


def invert_and_double_data(train_dataframes,test_dataframes):
    """
    testing hypothesis that data can be augmented for BERT by inverting labels
    """
    for i,(train_df,test_df) in enumerate(zip(train_dataframes,test_dataframes)):
        train_df_inverted = train_df.copy()
        test_df_inverted = test_df.copy()
        train_df_inverted["y"] = train_df_inverted["label"].map({"a1":1,"a2":0})
        test_df_inverted["y"] = test_df_inverted["label"].map({"a1":1,"a2":0})
        train_dataframes[i] = pd.concat(
            [
                train_dataframes[i],
                train_df_inverted
            ]
        )
        test_dataframes[i] = pd.concat(
            [
                test_dataframes[i],
                test_df_inverted
            ]
        )
    return train_dataframes,test_dataframes


def load_arg_pairs_UKP_IBMArg(data_source, N_folds=5, cross_topic_validation=False,bert_double_data=False,train_test_split=True):

    topics = DATASETS[data_source]["files"]
    data_dir = DATASETS[data_source]["data_dir"]

    df_all = pd.DataFrame()

    if cross_topic_validation:

        for topic, filename in topics:
            # print(filename)
            fpath = os.path.join(data_dir, filename)
            df_stance = pd.read_csv(fpath, sep="\t")

            df_stance["a1_id"] = df_stance["#id"].str.split("_").apply(lambda x: x[0])
            df_stance["a2_id"] = df_stance["#id"].str.split("_").apply(lambda x: x[1])

            df_stance["stance"] = topic.split("_")[1]
            df_stance["topic"] = topic.split("_")[0]
            df_stance["filename"] = filename
            df_stance["y"] = df_stance["label"].map({"a1": 0, "a2": 1})

            df_all = pd.concat([df_all, df_stance])

        train_dataframes, test_dataframes = get_cross_topic_validation_df(df_all)

    else:
        train_dataframes = [pd.DataFrame() for _ in itertools.repeat(None, N_folds)]
        test_dataframes = [pd.DataFrame() for _ in itertools.repeat(None, N_folds)]

        for topic, filename in topics:
            # print(filename)
            d = {}
            fpath = os.path.join(data_dir, filename)
            df_stance = pd.read_csv(fpath, sep="\t")

            df_stance["a1_id"] = df_stance["#id"].str.split("_").apply(lambda x: x[0])
            df_stance["a2_id"] = df_stance["#id"].str.split("_").apply(lambda x: x[1])

            df_stance["stance"] = topic.split("_")[1]
            df_stance["topic"] = topic.split("_")[0]
            df_stance["filename"] = filename

            df_stance["y"] = df_stance["label"].map({"a1": 0, "a2": 1})

            df_all = pd.concat([df_all, df_stance])

            skf = StratifiedKFold(n_splits=N_folds)
            for i, (train_indices, test_indices) in enumerate(
                skf.split(X=df_stance, y=df_stance["label"])
            ):

                train_dataframes[i] = pd.concat(
                    [train_dataframes[i], df_stance.iloc[train_indices]]
                )
                test_dataframes[i] = pd.concat(
                    [test_dataframes[i], df_stance.iloc[test_indices]]
                )

    print("Loaded {} arg pairs".format(data_source))

    if bert_double_data:
        train_dataframes, test_dataframes = invert_and_double_data(
            train_dataframes,test_dataframes
        )

    if not train_test_split:
        return df_all
    else:
        return train_dataframes, test_dataframes, df_all


def load_arg_pairs_IBM_Evi(N_folds=5, cross_topic_validation=False,bert_double_data=False,train_test_split=True):

    df_all = pd.DataFrame()
    for ftype in ["train", "test"]:
        fname = os.path.join(
            BASE_DATA_DIR, "IBM_Debater_(R)_EviConv-ACL-2019.v1/{}.csv".format(ftype)
        )
        df = pd.read_csv(fname)
        df = df.rename(
            columns={
                "evidence_1": "a1",
                "evidence_2": "a2",
                "evidence_1_id": "a1_id",
                "evidence_2_id": "a2_id",
                "evidence_1_detection_score": "a1_rank",
                "evidence_2_detection_score": "a2_rank",
            }
        )
        df["#id"] = "arg" + df["a1_id"].astype(str) + "_arg" + df["a2_id"].astype(str)
        df["label"] = df["label"].map({1: "a1", 2: "a2"})
        df["y"] = df["label"].map({"a1": 0, "a2": 1})

        df_all = pd.concat(
            [
                df_all,
                df[
                    [
                        "#id",
                        "label",
                        "y",
                        "a1",
                        "a2",
                        "topic",
                        "a1_id",
                        "a2_id",
                        "a1_rank",
                        "a2_rank",
                    ]
                ],
            ]
        )

    per_topic = df_all["topic"].value_counts()
    df_all = df_all[df_all["topic"].isin(per_topic[per_topic >= 50].index)]

    if not train_test_split:
        return df_all

    if cross_topic_validation:
        train_dataframes, test_dataframes = get_cross_topic_validation_df(df_all)

    else:
        train_dataframes = [pd.DataFrame() for _ in itertools.repeat(None, N_folds)]
        test_dataframes = [pd.DataFrame() for _ in itertools.repeat(None, N_folds)]

        for topic, df_topic in df_all.groupby("topic"):
            skf = StratifiedKFold(n_splits=N_folds)
            for i, (train_indices, test_indices) in enumerate(
                skf.split(X=df_topic, y=df_topic["label"])
            ):

                train_dataframes[i] = pd.concat(
                    [train_dataframes[i], df_topic.iloc[train_indices]]
                )
                test_dataframes[i] = pd.concat(
                    [test_dataframes[i], df_topic.iloc[test_indices]]
                )
    print("Loaded {} arg pairs".format("IBM_Evi"))

    if bert_double_data:
        train_dataframes, test_dataframes = invert_and_double_data(
            train_dataframes,test_dataframes
        )

    return train_dataframes, test_dataframes, df_all


def load_dalite_data(discipline, N_folds=5, cross_topic_validation=False, bert_double_data=False,train_test_split=True):

    data_dir = os.path.join(BASE_DIR, "tmp", "exp2", discipline, "all", "data_pairs")

    topics = os.listdir(data_dir)
    df_all = pd.DataFrame()
    for topic in topics:
        fpath = os.path.join(data_dir, topic)
        df_stance = pd.read_csv(fpath)
        # df_stance["discipline"] = topic.split("_")[0]
        df_stance["topic"] = topic
        df_all = pd.concat([df_all, df_stance])

    # if discipline:
    #     df_all = df_all[df_all["discipline"] == discipline]

    if not train_test_split:
        return df_all

    if cross_topic_validation:
        train_dataframes, test_dataframes = get_cross_topic_validation_df(df_all)
    else:

        train_dataframes = [pd.DataFrame() for _ in itertools.repeat(None, N_folds)]
        test_dataframes = [pd.DataFrame() for _ in itertools.repeat(None, N_folds)]

        for topic, df_topic in df_all.groupby("question"):
            skf = StratifiedKFold(n_splits=N_folds)
            for i, (train_indices, test_indices) in enumerate(
                skf.split(X=df_topic, y=df_topic["label"])
            ):

                train_dataframes[i] = pd.concat(
                    [train_dataframes[i], df_topic.iloc[train_indices]]
                )
                test_dataframes[i] = pd.concat(
                    [test_dataframes[i], df_topic.iloc[test_indices]]
                )
    print("Loaded {} arg pairs".format("dalite"))

    if bert_double_data:
        train_dataframes, test_dataframes = invert_and_double_data(
            train_dataframes,test_dataframes
        )

    return train_dataframes, test_dataframes, df_all


def load_arg_pairs(**kwargs):

    if kwargs["data_source"] in DATASETS.keys():
        return load_arg_pairs_UKP_IBMArg(**kwargs)
    else:
        if kwargs["data_source"] == "IBM_Evi":
            del kwargs["data_source"]
            return load_arg_pairs_IBM_Evi(**kwargs)
        else:
            del kwargs["data_source"]
            return load_dalite_data(**kwargs)


def filter_on_times_shown(_pairs_df,_df_topic):
    times_shown_counter = Counter()
    s = (
        _df_topic["rationales"]
        .dropna()
        .apply(
            lambda x: [
                int(k) for k in x.strip("[]").replace(" ", "").split(",") if k != ""
            ]
        )
    )
    _ = s.apply(lambda x: times_shown_counter.update(x))
    _df_topic["times_shown"]=_df_topic["id"].map(times_shown_counter)

    # filter out those not shown often enough
    df_topic = _df_topic[_df_topic["times_shown"]>=MIN_TIMES_SHOWN]

    ids=[f"arg{i}" for i in df_topic["id"].tolist()]

    pairs_df = _pairs_df[(
        (_pairs_df["a1_id"].isin(ids))
        |(_pairs_df["a2_id"].isin(ids))
    )].drop_duplicates(subset=["#id","label"])

    return pairs_df,df_topic

def get_topic_data(topic, discipline,output_dir):
    """
    given topic/question and associated discipline (needed for subdirectories),
    return mydalite answer observations, and associated pairs that are
    constructed using `make_pairs.py`

    filter on MIN_TIMES_SHOWN and MIN_WORD_COUNT

    Returns:
    ========
    Tuple of dataframes:
        - pairs_df
        - df_topic
    """

    data_dir_discipline=os.path.join(output_dir,"data")
    if discipline in DALITE_DISCIPLINES:
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

        # filter on times shown
        pairs_df_,df_topic_ = filter_on_times_shown(pairs_df,df_topic)

        # filter on min word count
        df_topic_ = df_topic_[df_topic_["rationale"].str.count("\w+")>=MIN_WORD_COUNT]
        ids=[f"arg{i}" for i in df_topic_["id"].tolist()]
        pairs_df_ = pairs_df_[(
            (pairs_df_["a1_id"].isin(ids))
            |(pairs_df_["a2_id"].isin(ids))
        )].drop_duplicates(subset=["#id","label"])

    else:
        # load pairs
        pairs_df_ = pd.read_csv(
            os.path.join(
                "{}_pairs".format(data_dir_discipline), "{}.csv".format(topic)
            ),
            sep="\t"
        )
        pairs_df_["a1_id"] = pairs_df_["#id"].str.split("_").apply(lambda x: x[0])
        pairs_df_["a2_id"] = pairs_df_["#id"].str.split("_").apply(lambda x: x[1])
        pairs_df_ = pairs_df_[pairs_df_["a1_id"] != pairs_df_["a2_id"]]

        # make df of just the individual arguments with their id's from the pairs
        df_topic_=pd.concat([
            pairs_df_[["a1_id","a1"]].rename(columns={"a1_id":"id","a1":"rationale"}).drop_duplicates("id"),
            pairs_df_[["a2_id","a2"]].rename(columns={"a2_id":"id","a2":"rationale"}).drop_duplicates("id")
        ]).drop_duplicates("id")
        df_topic_["transition"]="-"
        pairs_df_["transition"]="-"
        df_topic_["topic"]=topic

    return pairs_df_, df_topic_


def get_discipline_data(discipline,output_dir_name,population):
    """
    Load all data, pairs and answers,
    after filtering on MIN_TIMES_SHOWN, and MIN_WORD_COUNT,
    only keep topics which have MIN_ANSWERS & MIN_PAIRS

    Returns:
    ========
        pairs_df_all,df_topics_all

    """
    pairs_df_all=pd.DataFrame()
    df_topics_all = pd.DataFrame()
    data_dir_discpline = os.path.join(
            BASE_DIR, "tmp", output_dir_name, discipline, population, "data"
        )
    output_dir = os.path.join(data_dir_discpline, os.pardir)
    topics = os.listdir(data_dir_discpline)

    for t, topic in enumerate(topics):
        if t%(len(topics)//10)==0:
            print(f"\t{t}/{len(topics)}")
        topic=topic.replace(".csv","")
        pairs_df,df_topic = get_topic_data(topic,discipline,output_dir)

        if (
            (
                df_topic.shape[0]>MIN_ANSWERS
            ) or (
                discipline not in DALITE_DISCIPLINES
            )
        ):
            df_topic["surface_n_words"] = df_topic["rationale"].str.count("\w+")

            # what is difference in WC for each pair?
            pairs_df["topic"]=topic
            pairs_df["wc_diff"]=(
                pairs_df["a1"].str.count("\w+")
                -pairs_df["a2"].str.count("\w+")
            ).abs()

            df_topics_all=pd.concat([df_topics_all,df_topic])
            pairs_df_all=pd.concat([pairs_df_all,pairs_df])

        else:
            if discipline in DALITE_DISCIPLINES:
                print(f"\t\t skip {t}: {df_topic.shape[0]} answers; {pairs_df.shape[0]} pairs; {topic}")

    return pairs_df_all,df_topics_all


# ARCHIVE
def load_textbook_data():

    data_dir = "gdrive/My Drive/Colab Notebooks/convincingness/data/textbooks"

    # Term Document Matrices for Each Discipline in Dalite
    vec_by_subject = {}

    openstax_textbook_disciplines = {
        "Chemistry": ["chemistry-2e"],
        "Biology": ["biology-2e"],
        "Physics": [
            "university-physics-volume-3",
            "university-physics-volume-2",
            "university-physics-volume-1",
        ],
        "Statistics": ["introductory-statistics"],
    }
    for subject in openstax_textbook_disciplines.keys():
        books = openstax_textbook_disciplines[subject]
        keywords = {}

        files_all = []
        for book in books:
            book_dir = os.path.join(data_dir, book)
            filenames = [os.path.join(book_dir, fn) for fn in os.listdir(book_dir)]
            files_all.extend(filenames)

        vec_subject = CountVectorizer(input="filename", stop_words="english")
        vec_subject.fit(files_all)

        vec_by_subject[subject] = vec_subject
    return vec_by_subject
