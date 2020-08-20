from __future__ import unicode_literals
import os
import json
import re
import spacy
import plac
import datetime
import data_loaders

from spacy.matcher import PhraseMatcher
from spacy_readability import Readability
import pandas as pd
from utils_scrape_openstax import OPENSTAX_TEXTBOOK_DISCIPLINES

from make_pairs import make_pairs_by_topic, filter_out_stick_to_own

from argBT import get_rankings_baseline, get_rankings

# make sure this matches the calculations below
PRE_CALCULATED_FEATURES = {
    "surface": [
        "rationale_word_count",
        # "num_sents",
    ],
    "syntax": [],
    "readability": ["flesch_kincaid_grade_level", "flesch_kincaid_reading_ease",],
    "lexical": ["num_equations", "num_keywords"],
    "convincingness": ["convincingness_BT", "convincingness_baseline",],
    "semantic": [],
}


def extract_surface_features(df):
    surface_features = {}

    df["rationale_word_count"] = df["rationale"].str.count("\w+")

    surface_features["rationale_word_count"] = [
        [d["id"], d["rationale_word_count"]]
        for d in df[["id", "rationale_word_count"]].to_dict(orient="records")
    ]

    return surface_features


def on_match(matcher, doc, id, matches):
    """
    call back function to be executed everytime a phrase is matched
    inside `get_matcher` function below
    """

    for m in matches:
        print(doc[m[1] - 2 : m[2] + 2])


def get_matcher(subject, nlp, on_match=None):
    """
    given a subject and nlp object,
    return a phrase mather object that has patterns from
    OpenStax textbook of that discipline
    """
    # print(subject)
    books = OPENSTAX_TEXTBOOK_DISCIPLINES[subject]
    keywords = {}
    for book in books:
        # print(book)
        book_dir = os.path.join(
            data_loaders.BASE_DIR, os.pardir, "textbooks", subject, book
        )
        files = os.listdir(book_dir)
        keyword_files = [f for f in files if "key-terms" in f]
        for fn in keyword_files:
            # print(fn)
            with open(os.path.join(book_dir, fn), "r") as f:
                keywords.update(json.load(f))

    keywords_sorted = list(keywords.keys())
    keywords_sorted.sort()

    matcher = PhraseMatcher(nlp.vocab, attr="lower")
    patterns = [nlp.make_doc(text) for text in keywords_sorted]

    matcher.add("KEY_TERMS", patterns, on_match=on_match)

    return matcher


def extract_lexical_features(rationales, subject, nlp):
    """
    given array of rationales,
    return dict of arrays holding lexical features for each, including:
        - number of keywords
        - number of equations
    """
    lexical_features = {}
    matcher = get_matcher(subject=subject, nlp=nlp)
    try:
        lexical_features["num_keywords"] = list(
            zip(
                rationales["id"],
                [
                    len(
                        set(
                            [
                                str(doc[start:end])
                                for match_id, start, end in matcher(doc)
                            ]
                        )
                    )
                    for doc in nlp.pipe(rationales["rationale"], batch_size=50)
                ],
            )
        )
    except TypeError:
        pass

    eqn_re = re.compile(r"([\w\/^\*\.+-]+\s?=\s?[\w\/^\*\.+-]+)")
    lexical_features["num_equations"] = list(
        zip(
            rationales["id"],
            pd.Series(rationales["rationale"]).str.count(eqn_re).to_list(),
        )
    )

    return lexical_features


def extract_syntactic_features(rationales, nlp):
    """
    given array of rationales,
    return dict of arrays holding synctatic features for each, including:
        - num_sents
        - num_verbs
        - num_conj
    """

    syntactic_features = {}

    syntactic_features["num_sents"] = list(
        zip(
            rationales["id"],
            [
                len(list(doc.sents))
                for doc in nlp.pipe(rationales["rationale"], batch_size=50)
            ],
        )
    )

    syntactic_features["num_verbs"] = list(
        zip(
            rationales["id"],
            [
                len([token.text for token in doc if token.pos_ == "VERB"])
                for doc in nlp.pipe(rationales["rationale"], batch_size=50)
            ],
        )
    )

    syntactic_features["num_conj"] = list(
        zip(
            rationales["id"],
            [
                len(
                    [
                        token.text
                        for token in doc
                        if token.pos_ == "CCONJ" or token.pos_ == "SCONJ"
                    ]
                )
                for doc in nlp.pipe(rationales["rationale"], batch_size=50)
            ],
        )
    )
    return syntactic_features


def extract_readability_features(rationales, nlp):
    """
    given array of rationales,
    return dict of arrays holding synctatic features for each, including:
        - flesch_kincaid_grade_level
        - flesch_kincaid_reading_ease
        - dale_chall
        - automated_readability_index
        - coleman_liau_index
    """
    nlp.add_pipe(Readability())
    readability_features_list = [
        ("flesch_kincaid_grade_level"),
        ("flesch_kincaid_reading_ease"),
        # ("dale_chall"),
        # ("automated_readability_index"),
        # ("coleman_liau_index"),
    ]

    readability_features = {}

    for f in readability_features_list:
        readability_features[f] = list(
            zip(
                rationales["id"],
                [
                    round(getattr(doc._, f), 3)
                    for doc in nlp.pipe(rationales["rationale"], batch_size=50)
                ],
            )
        )
    return readability_features


def extract_convincingness_features(topic,discipline):

    data_dir_discipline = os.path.join(data_loaders.BASE_DIR,"tmp","switch_exp",discipline)

    fp=os.path.join(data_dir_discipline,"data_pairs","pairs_{}.csv".format(topic.replace("/", "_")))
    df_pairs = pd.read_csv(fp)
    convincingness_features = {}

    for f in PRE_CALCULATED_FEATURES["convincingness"]:
        if f == "convincingness_BT":
            r = get_rankings(df_pairs)[1]
        else:
            fp = os.path.join(data_dir_discipline,"data","{}.csv".format(topic.replace("/", "_")))
            df_topic = pd.read_csv(fp)
            r = get_rankings_baseline(df_topic)[1]

        # remove "arg" prefix
        r = {int(k[3:]): v for k, v in r.items()}

        # store as list of lists, where first element is answer id, and second
        # is feature value
        convincingness_features[f] = list(r.items())

    return convincingness_features


def extract_features_and_save(
    with_features_dir, df_answers, topic, feature_type, nlp, subject=None
):
    """

    """
    feature_type_fpath = os.path.join(
        with_features_dir, topic + "_" + feature_type + ".json"
    )

    if os.path.exists(feature_type_fpath):
        # print(feature_type + " features already calculated")
        with open(feature_type_fpath, "r") as f:
            features = json.load(f)
        return features

    else:
        print("\t\t\tcalculating features " + feature_type)
        if feature_type == "syntax":
            features = extract_syntactic_features(
                df_answers[["id", "rationale"]], nlp=nlp
            )
        elif feature_type == "readability":
            features = extract_readability_features(
                df_answers[["id", "rationale"]], nlp=nlp
            )
        elif feature_type == "lexical":
            features = extract_lexical_features(
                df_answers[["id", "rationale"]], nlp=nlp, subject=subject,
            )
        elif feature_type == "convincingness":
            features = extract_convincingness_features(
                topic=topic, discipline= subject
            )
        with open(feature_type_fpath, "w") as f:
            json.dump(features, f, indent=2)

        return features


def get_features(
    df_answers, path_to_data, topic, subject=None,
):
    """
    append lexical, syntactic or readability features for rationales
    """

    nlp = spacy.load("en_core_web_sm")

    df_answers.loc[df_answers["rationale"].isna(), "rationale"] = " "

    with_features_dir = os.path.join(path_to_data, topic + "_features")

    if not os.path.exists(with_features_dir):
        os.mkdir(with_features_dir)

    # TO DO: "lexical"
    for feature_type in ["convincingness"]:  # , "syntax", "readability"]:
        features = extract_features_and_save(
            with_features_dir=with_features_dir,
            df_answers=df_answers,
            topic=topic,
            feature_type=feature_type,
            nlp=nlp,
            subject=subject,
        )
        for f in features:
            df_answers = pd.merge(
                df_answers,
                pd.DataFrame(features[f], columns=["id", f]),
                on="id",
                how="left",
            )

    return df_answers


def main(discipline):
    print(discipline)
    print("Start: {}".format(datetime.datetime.now()))

    RESULTS_DIR = os.path.join(data_loaders.BASE_DIR, "tmp", "switch_exp")

    data_dir_discipline = os.path.join(RESULTS_DIR, discipline, "data")

    # sort files by size to get biggest ones done first
    # https://stackoverflow.com/a/20253803
    all_files = (
        os.path.join(data_dir_discipline, filename)
        for basedir, dirs, files in os.walk(data_dir_discipline)
        for filename in files
    )
    all_files = sorted(all_files, key=os.path.getsize, reverse=True)

    topics = [os.path.basename(fp)[:-4] for fp in all_files]

    results_dir_discipline = os.path.join(RESULTS_DIR, discipline, "data_with_features")
    topics_already_done = [fp[:-5] for fp in os.listdir(results_dir_discipline)]

    topics_to_do = [t for t in topics if t not in topics_already_done]

    df_all = pd.DataFrame()

    for t, topic in enumerate(topics_to_do):

        print("{}/{}: {}".format(t, len(topics_to_do), topic))
        df_topic = pd.read_csv(
            os.path.join(data_dir_discipline, "{}.csv".format(topic))
        )

        df_topic = get_features(
            df_answers=df_topic,
            topic=topic,
            path_to_data=results_dir_discipline,
            subject=discipline,
        )

        df_all = pd.concat([df_all, df_topic])

    fp = os.path.join(results_dir_discipline, "all_topics_with_features.csv")
    df_all.to_csv(fp)
    print("Finished: {} ".format(datetime.datetime.now()))


if __name__ == "__main__":
    import plac

    plac.call(main)
