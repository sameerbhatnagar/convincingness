from __future__ import unicode_literals
import os
import json
import re
import spacy
import plac
import datetime
import data_loaders
import pathlib
import numpy as np

from spacy.matcher import PhraseMatcher
from spacy_readability import Readability

from spacy.symbols import (
    ADJ,
    ADP,
    ADV,
    AUX,
    CONJ,
    CCONJ,
    NOUN,
    NUM,
    PRON,
    SCONJ,
    SYM,
    VERB,
    X,
)

import pandas as pd
from utils_scrape_openstax import OPENSTAX_TEXTBOOK_DISCIPLINES

from make_pairs import make_pairs_by_topic, filter_out_stick_to_own

from argBT import RESULTS_DIR, get_rankings_winrate, get_rankings_elo, get_topic_data

nlp = spacy.load("en_core_web_sm")

from spellchecker import SpellChecker

DROPPED_POS = ["PUNCT", "SPACE"]

# make sure this matches the calculations below
PRE_CALCULATED_FEATURES = {
    "surface": ["rationale_word_count",],
    "syntax": [],
    "readability": ["flesch_kincaid_grade_level", "flesch_kincaid_reading_ease",],
    "lexical": ["n_equations", "n_spelling_errors", "n_keywords"],
    "convincingness": ["convincingness_BT", "convincingness_baseline",],
    "semantic": [],
}

EQUATION_TAG = "EQUATION"
EXPRESSION_TAG = "EXPRESSION"

fp = os.path.join(
    RESULTS_DIR, os.pardir, os.pardir, os.pardir, "all_questions_physics.csv"
)
df_q = pd.read_csv(fp)


def get_topic_data_cleaned(topic, discipline):
    """
    get answers for question (topic), but clean up rationales by
    replacing all expressions and equations with common tags
    """

    eqn_re = re.compile(r"([\w\/^\*\.\(\)+-]+\s?[=]\s?[\w\/^\*\.\(\)+-]+)")
    expr_re = re.compile(r"([\w\/^\*\.\(\)+-]+\s?[+\*\-/]\s?[\w\/^\*\.\(\)+-]+)")

    _, df = get_topic_data(topic=topic, discipline=discipline)
    df["rationale"] = (
        df["rationale"]
        .fillna(" ")
        .str.replace(eqn_re, EQUATION_TAG)
        .str.replace(expr_re, EXPRESSION_TAG)
    )
    rationales = df[["rationale", "id"]].values
    return rationales


def extract_surface_features(topic, discipline):

    rationales = get_topic_data_cleaned(topic, discipline)

    surface_features = {}

    surface_features["n_words"] = {
        arg_id: len([token for token in doc if token.pos_ not in DROPPED_POS])
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    surface_features["n_content_words"] = {
        arg_id: len(
            [
                token
                for token in doc
                if token.pos_ not in DROPPED_POS and not token.is_stop
            ]
        )
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    surface_features["TTR"] = {
        arg_id: len(set([token.text for token in doc if token.pos_ not in DROPPED_POS]))
        / (len([token.text for token in doc if token.pos_ not in DROPPED_POS])+1)
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    surface_features["TTR_content"] = {
        arg_id: len(
            set(
                [
                    token.text
                    for token in doc
                    if token.pos_ not in DROPPED_POS and not token.is_stop
                ]
            )
        )
        / (len(
            [
                token.text
                for token in doc
                if token.pos_ not in DROPPED_POS and not token.is_stop
            ]
        )+1)
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    surface_features["n_sents"] = {
        arg_id: len([sent for sent in doc.sents])
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    return surface_features


def on_match(matcher, doc, id, matches):
    """
    call back function to be executed everytime a phrase is matched
    inside `get_matcher` function below
    """

    for m in matches:
        print(doc[m[1] - 2 : m[2] + 2])


def get_matcher(subject, nlp, topic=None, on_match=None):
    """
    given a subject and nlp object,
    return a phrase mather object that has patterns from
    OpenStax textbook of that discipline
    """

    if topic:
        texts = df_q.loc[df_q["title"].str.startswith(topic), ["text", "expert_rationale"]].values
        terms = []
        for text in texts[0]:
            if type(text)!=float:
                doc = nlp(text)
                terms.extend(
                    [chunk.text for chunk in doc.noun_chunks]
                    + [
                        token.text
                        for token in doc
                        if not token.is_stop and token.pos_ not in DROPPED_POS
                    ]
                )
        keywords_sorted = list(set(terms))

    else:
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


def extract_lexical_features(topic, discipline):
    """
    given array of rationales,
    return dict of arrays holding lexical features for each, including:
        - number of keywords
        - number of equations
    """

    rationales = get_topic_data_cleaned(topic, discipline)

    lexical_features = {}
    matcher = get_matcher(subject=discipline, nlp=nlp)
    try:
        lexical_features["n_keyterms"] = {
            arg_id: len(
                set([str(doc[start:end]) for match_id, start, end in matcher(doc)])
            )
            for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
        }

    except TypeError:
        pass

    matcher_prompt = get_matcher(subject=discipline, topic=topic, nlp=nlp)
    lexical_features["n_prompt_terms"] = {
        arg_id: len(
            set([str(doc[start:end]) for match_id, start, end in matcher_prompt(doc)])
        )
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    lexical_features["n_equations"] = {
        arg_id: len(
            [
                token.text
                for token in doc
                if token.text == EQUATION_TAG or token.text == EXPRESSION_TAG
            ]
        )
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    spell = SpellChecker()
    lexical_features["n_spelling_errors"] = {
        arg_id: len(
            spell.unknown(
                [token.text for token in doc if token.pos_ not in DROPPED_POS]
            )
        )
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    return lexical_features


def extract_syntactic_features(topic, discipline):
    """
    given array of rationales,
    return dict of arrays holding synctatic features for each, including:
        - num_sents
        - num_verbs
        - num_conj
    """
    rationales = get_topic_data_cleaned(topic, discipline)

    syntactic_features = {}

    POS_TAGS = [ADJ, ADP, ADV, AUX, CONJ, CCONJ, NOUN, NUM, PRON, SCONJ, SYM, VERB, X]
    for pos_tag in POS_TAGS:
        feature_name = "n_{}".format(pos_tag)

        syntactic_features[feature_name] = {
            arg_id: len([token for token in doc if token.pos == pos_tag])
            for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
        }

    syntactic_features["n_negations"] = {
        arg_id: len([token for token in doc if token.dep_ == "neg"])
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    syntactic_features["n_VERB_mod"] = {
        arg_id: len([token for token in doc if token.tag_ == "MD"])
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    syntactic_features["n_PRON_pers"] = {
        arg_id: len([token for token in doc if token.tag_ == "PRP"])
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    return syntactic_features


def extract_readability_features(topic, discipline):
    """
    given array of rationales,
    return dict of arrays holding synctatic features for each, including:
        - flesch_kincaid_grade_level
        - flesch_kincaid_reading_ease
        - dale_chall
        - automated_readability_index
        - coleman_liau_index
    """

    rationales = get_topic_data_cleaned(topic, discipline)

    read = Readability()
    # nlp.add_pipe(read, last=True)

    readability_features_list = [
        ("flesch_kincaid_grade_level"),
        ("flesch_kincaid_reading_ease"),
        ("dale_chall"),
        ("automated_readability_index"),
        ("coleman_liau_index"),
    ]

    readability_features = {}

    for f in readability_features_list:
        readability_features[f] = {
            arg_id: round(getattr(doc._, f), 3)
            for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
        }

    return readability_features


def extract_convincingness_features(topic, discipline, timestep=None):
    """
    This function retunr sthe same data structure as with the other
    `extract_<>_features` functions, except that teh optional `timestep`
    argument allows to learn rankings as they change over over time.
    Arguments:
    =========
        `topic` (question title), `discipline`, are strings which help load the
        data from the correct csv files
        `timestep` is optional argument, which will filter the data to train
        rank scores only based on upto a certain number of students.

    Returns:
    ========
        dict, where the keyes are the different measures of convincingness
    """

    from switch_exp import MIN_TRAINING_RECORDS

    data_dir_discipline = os.path.join(
        data_loaders.BASE_DIR, "tmp", "fine_grained_arg_rankings", discipline
    )

    fp = os.path.join(
        data_dir_discipline,
        "data_pairs",
        "pairs_{}.csv".format(topic.replace("/", "_")),
    )
    df_pairs = pd.read_csv(fp)
    if timestep:
        df_pairs = df_pairs[df_pairs["annotation_rank_by_time"] < timestep]
        if df_pairs.shape[0] < MIN_TRAINING_RECORDS:
            return {f: [] for f in PRE_CALCULATED_FEATURES["convincingness"]}

    convincingness_features = {}

    for f in PRE_CALCULATED_FEATURES["convincingness"]:
        if f == "convincingness_BT":
            r = get_rankings(df_pairs)[1]
        else:
            fp = os.path.join(
                data_dir_discipline, "data", "{}.csv".format(topic.replace("/", "_"))
            )
            df_topic = pd.read_csv(fp)
            if timestep:
                df_topic = df_topic[df_topic["a_rank_by_time"] < timestep]
            r = get_rankings_baseline(df_topic)[1]

        # remove "arg" prefix
        r = {int(k.replace("arg", "")): v for k, v in r.items()}

        # store as list of lists, where first element is answer id, and second
        # is feature value
        convincingness_features[f] = list(r.items())

    return convincingness_features


def extract_features_and_save(discipline, topic, feature_type):
    """
    dispatch to correct function based on feature type, save result, and return
    feature_dict
    """

    feature_type_fpath = os.path.join(
        RESULTS_DIR,
        discipline,
        "data_with_features",
        feature_type,
        "{}.json".format(topic),
    )

    print("\t\t\tcalculating features " + feature_type)
    if feature_type == "syntax":
        features = extract_syntactic_features(topic=topic, discipline=discipline)
    elif feature_type == "readability":
        features = extract_readability_features(topic=topic, discipline=discipline)
    elif feature_type == "lexical":
        features = extract_lexical_features(topic=topic, discipline=discipline)
    elif feature_type == "convincingness":
        features = extract_convincingness_features(topic=topic, discipline=subject)
    elif feature_type == "surface":
        features = extract_surface_features(topic=topic, discipline=discipline)
    with open(feature_type_fpath, "w") as f:
        json.dump(features, f, indent=2)

    return features


def get_features(topic, discipline, feature_type):
    """
    append features onto df_answers
    """

    # TO DO: "lexical"
    if feature_type == "all":
        feature_types = ["surface", "lexical", "syntax", "readability"]
    else:
        feature_types = [feature_type]

    _, df_answers = get_topic_data(topic=topic, discipline=discipline)

    for feature_type in feature_types:
        features = extract_features_and_save(
            topic=topic, discipline=discipline, feature_type=feature_type,
        )
        for f in features:
            df_answers = pd.merge(
                df_answers,
                pd.DataFrame(features[f], columns=["id", f]),
                on="id",
                how="left",
            )

    return df_answers


def append_features(topic, discipline, feature_types_included, timestep=None):
    """
    Arguments:
    =========
        df: dataframe with answers from peer instruction, with column "id"
            for each answer, the topic, and discipline
        feature_types_included: string from one of the keys of
            PRE_CALCULATED_FEATURES defined at top of this file
    Returns:
    ========
        df: dataframe with feature columns appended
    """
    from argBT import get_data_dir

    data_dir_discipline = get_data_dir(discipline)

    _, df = get_topic_data(topic, discipline)

    # append columns with pre-calculated features
    features_dir = os.path.join(RESULTS_DIR, discipline, "data_with_features")
    if timestep:
        # get the other features normally
        feature_types_included = [
            f for f in feature_types_included if f != "convincingness"
        ]
        # get the convincingness features as calculated with the data before current timestep
        rank_score_types = ["baseline", "BT"]
        for rank_score_type in rank_score_types:
            colname = "convincingness_{}".format(rank_score_type)
            rank_scores_list = extract_convincingness_features(
                topic=topic, discipline=discipline, timestep=timestep
            )[colname]
            rank_scores = pd.DataFrame(rank_scores_list, columns=["id", colname])

            df = pd.merge(df, rank_scores, on="id", how="left")

    for feature_type in feature_types_included:
        feature_type_fpath = os.path.join(
            features_dir, feature_type, "{}.json".format(topic)
        )
        with open(feature_type_fpath, "r") as f:
            features_json = json.load(f)

        # each feature_type includes multiple features
        for feature in features_json:
            df = pd.merge(
                df,
                pd.DataFrame(features_json[feature], columns=["id", feature]),
                on="id",
                how="left",
            )
    return df


def main(
    discipline: (
        "Discipline",
        "positional",
        None,
        str,
        ["Physics", "Biology", "Chemistry"],
    ),
    feature_type: (
        "Feature Type",
        "positional",
        None,
        str,
        ["all", "surface", "readability", "lexical", "syntax"],
    ),
    largest_first: ("Largest Files First", "flag", "l", bool,),
):
    print("{} - {}".format(discipline, feature_type))
    print("Start: {}".format(datetime.datetime.now()))

    results_sub_dir = os.path.join(
        RESULTS_DIR, discipline, "data_with_features", feature_type
    )

    # make directory if doesn't exist
    pathlib.Path(results_sub_dir).mkdir(parents=True, exist_ok=True)

    data_dir_discipline = os.path.join(RESULTS_DIR, discipline, "data")

    # sort files by size to get biggest ones done first
    # https://stackoverflow.com/a/20253803
    all_files = (
        os.path.join(data_dir_discipline, filename)
        for basedir, dirs, files in os.walk(data_dir_discipline)
        for filename in files
    )
    all_files = sorted(all_files, key=os.path.getsize, reverse=largest_first)

    topics = [os.path.basename(fp)[:-4] for fp in all_files]

    topics_already_done = [fp[:-5] for fp in os.listdir(results_sub_dir)]

    topics_to_do = [t for t in topics if t not in topics_already_done]

    df_all = pd.DataFrame()

    for t, topic in enumerate(topics_to_do):

        print("{}/{}: {}".format(t, len(topics_to_do), topic))

        df_topic = get_features(
            topic=topic, discipline=discipline, feature_type=feature_type,
        )

        df_all = pd.concat([df_all, df_topic])

    fp = os.path.join(results_sub_dir, "all_topics_with_features.csv")
    df_all.to_csv(fp)
    print("Finished: {} ".format(datetime.datetime.now()))


if __name__ == "__main__":
    import plac

    plac.call(main)
    plac.call(main)
