from __future__ import unicode_literals
import os
import json
import re
import spacy

import data_loaders

from spacy.matcher import PhraseMatcher
from spacy_readability import Readability
import pandas as pd
from utils_scrape_openstax import OPENSTAX_TEXTBOOK_DISCIPLINES


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
        with open(feature_type_fpath, "w") as f:
            json.dump(features, f, indent=2)

        return features


def get_features(df_answers,path_to_data,topic, subject=None,
):
    """
    append lexical, syntactic or readability features for rationales
    """

    nlp = spacy.load("en_core_web_md")


    df_answers["rationale"] = df_answers["rationale"].fillna(" ")

    with_features_dir = os.path.join(path_to_data, topic + "_features")

    if not os.path.exists(with_features_dir):
        os.mkdir(with_features_dir)

    # TO DO: "lexical"
    for feature_type in ["syntax", "readability"]:
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
