import pandas as pd
import os
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_DIR = os.path.join(BASE_DIR, "data")
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

    df_all = df_all.rename(columns={"question": "topic"})
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


def load_arg_pairs_UKP_IBMArg(data_source, N_folds=5, cross_topic_validation=False):

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

            skf = StratifiedKFold(n_splits=N_folds,random_state=123)
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
    return train_dataframes, test_dataframes, df_all


def load_arg_pairs_IBM_Evi(N_folds=5, cross_topic_validation=False):

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

    return train_dataframes, test_dataframes, df_all


def load_dalite_data(N_folds=5, cross_topic_validation=False, discipline=None):

    data_dir = os.path.join(BASE_DATA_DIR, "mydalite_arg_pairs")

    topics = os.listdir(data_dir)
    df_all = pd.DataFrame()
    for topic in topics:
        fpath = os.path.join(data_dir, topic)
        df_stance = pd.read_csv(fpath)
        df_stance["discipline"] = topic.split("_")[0]
        df_stance["question"] = topic.split("_")[1]
        df_all = pd.concat([df_all, df_stance])

    if cross_topic_validation:
        if discipline:
            df_all = df_all[df_all["discipline"] == discipline]

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
