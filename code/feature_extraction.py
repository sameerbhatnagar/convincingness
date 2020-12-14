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
from scipy.spatial.distance import cosine

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
    PUNCT,
    SPACE,
)

import pandas as pd
from utils_scrape_openstax import OPENSTAX_TEXTBOOK_DISCIPLINES

from make_pairs import make_pairs_by_topic, filter_out_stick_to_own

from argBT import get_rankings_winrate, get_rankings_elo, get_topic_data

from plots import BASE_DIR

from make_pairs import EQUATION_TAG, EXPRESSION_TAG, OOV_TAG

nlp = spacy.load("en_core_web_md", disable=["ner"])

from spellchecker import SpellChecker

DROPPED_POS = [PUNCT, SPACE, X]

DALITE_DISCIPLINES = ["Physics","Chemistry","Ethics"]


import html as ihtml
from bs4 import BeautifulSoup

# https://gist.github.com/drussellmrichie/47deb429350e2e99ffb3272ab6ab216a
def tree_height(root):
    """
    Find the maximum depth (height) of the dependency parse of a spacy sentence by starting with its root
    Code adapted from https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
    :param root: spacy.tokens.token.Token
    :return: int, maximum height of sentence's dependency parse tree
    """
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)


# https://www.kaggle.com/ceshine/remove-html-tags-using-beautifulsoup
def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text)).text
    # text = re.sub(r"http[s]?://\S+", "", text)
    # text = re.sub(r"\s+", " ", text)
    return text


def get_questions_df(discipline):

    if discipline in ["Physics", "Chemistry"]:
        fp = os.path.join(BASE_DIR, "all_questions.csv")
        df_q = pd.read_csv(fp)

    elif discipline == "Ethics":

        data_dir = os.path.join(BASE_DIR, "data_harvardx")

        fp = os.path.join(data_dir, "dalite_20161101.csv")
        df = pd.read_csv(fp)
        df_q1 = df[["assignment_id", "question_id", "question_text"]].drop_duplicates(
            ["assignment_id", "question_id"]
        )

        files = [
            f
            for f in os.listdir(os.path.join(data_dir, "video-text"))
            if not f.startswith(".") and "post" not in f
        ]
        results = []
        for fn in files:
            d = {}
            d["assignment_id"] = fn.replace(".txt", "").split("_")[0]
            fp = os.path.join(data_dir, "video-text", fn)
            keyname = "text"
            with open(fp, "r") as f:
                d[keyname] = f.read()
            results.append(d)

        df_q = pd.DataFrame(results)

        files = [
            f
            for f in os.listdir(os.path.join(data_dir, "video-text"))
            if not f.startswith(".") and "post" in f
        ]
        results = []
        for fn in files:
            d = {}
            d["assignment_id"] = fn.replace(".txt", "").split("_")[0]
            fp = os.path.join(data_dir, "video-text", fn)
            keyname = "expert_rationale"
            with open(fp, "r") as f:
                d[keyname] = f.read()
            results.append(d)

        df_q = df_q.merge(pd.DataFrame(results), on="assignment_id").sort_values(
            "assignment_id"
        )
        df_q["text"] = (
            df_q["text"]
            .str.replace("\[MUSIC\]", "")
            .str.replace("\[...\]", "")
            .str.replace("SPEAKER: ", "")
            .str.replace("SPEAKER 1: ", "")
            .str.replace("PROFESSOR: ", "")
            .str.replace("\[Music\]", "")
        )

        df_q["expert_rationale"] = (
            df_q["expert_rationale"]
            .str.replace("MICHAEL SANDEL: ", "")
            .str.replace("MICHEAL SANDEL: ", "")
            .str.replace("MICHAEL J. SANDEL: ", "")
            .str.replace("PROF. Michael Sandel: ", "")
            .str.replace("Professor Sandel: ", "")
            .str.replace("PROFESSOR: ", "")
            .str.replace("SPEAKER 1: ", "")
            .str.replace("SPEAKER: ", "")
        )

        df_q = df_q.merge(df_q1, on="assignment_id", how="outer")
        df_q["text"] = (
            df_q["text"].astype(str) + " " + df_q["question_text"].astype(str)
        )

        df_q.loc[df_q["question_text"].isna(), "question_text"] = df_q.loc[
            df_q["question_text"].isna(), "text"
        ]
        df_q["question_id"] = df_q["question_id"].fillna(0).astype(int)

        df_q["topic"] = (
            df_q["question_text"]
            .str.strip("[?.,]")
            .apply(lambda x: max(x.split(), key=len))
        )
        df_q["title"] = df_q["question_id"].astype(str) + "_" + df_q["topic"]

    df_q = df_q.fillna(" ")

    df_q["text"] = df_q["text"].apply(clean_text)

    df_q["expert_rationale"] = df_q["expert_rationale"].apply(clean_text)
    return df_q


def extract_surface_features(topic, discipline, output_dir):
    """
    """
    feature_type = "surface"
    _, df = get_topic_data(topic, discipline, output_dir)
    rationales = df[["rationale", "id"]].values

    surface_features = {}

    surface_features["{}_n_words".format(feature_type)] = {
        arg_id: len([token for token in doc if token.pos not in DROPPED_POS])
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    surface_features["{}_n_content_words".format(feature_type)] = {
        arg_id: len(
            [
                token
                for token in doc
                if token.pos not in DROPPED_POS
                and not token.is_stop
                and token.text != OOV_TAG
            ]
        )
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    surface_features["{}_TTR".format(feature_type)] = {
        arg_id: len(set([token.text for token in doc if token.pos not in DROPPED_POS]))
        / (len([token.text for token in doc if token.pos not in DROPPED_POS]) + 1)
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    surface_features["{}_TTR_content".format(feature_type)] = {
        arg_id: len(
            set(
                [
                    token.text
                    for token in doc
                    if token.pos not in DROPPED_POS
                    and not token.is_stop
                    and token.text != OOV_TAG
                ]
            )
        )
        / (
            len(
                [
                    token.text
                    for token in doc
                    if token.pos not in DROPPED_POS
                    and not token.is_stop
                    and token.text != OOV_TAG
                ]
            )
            + 1
        )
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    surface_features["{}_n_sents".format(feature_type)] = {
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

    df_q = get_questions_df(discipline=subject)

    if topic:
        texts = df_q.loc[
            df_q["title"].str.startswith(topic), ["text", "expert_rationale"]
        ].values
        terms = []
        for text in texts[0]:
            if type(text) != float:
                doc = nlp(text)
                terms.extend(
                    [chunk.text for chunk in doc.noun_chunks]
                    + [
                        token.text
                        for token in doc
                        if not token.is_stop and token.pos not in DROPPED_POS
                    ]
                )
        keywords_sorted = list(set(terms))

    else:
        if subject == "same_teacher_two_groups":
            subject = "Physics"
        books = OPENSTAX_TEXTBOOK_DISCIPLINES[subject]
        keywords_dict = {}
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
                    keywords_dict.update(json.load(f))

        keywords = list(keywords_dict.keys())
        kt = [
            [
                token.text
                for token in doc
                if token.pos not in DROPPED_POS and not token.is_stop
            ]
            for doc in nlp.pipe(keywords, batch_size=20)
        ]
        for k in kt:
            keywords.extend(k)
        keywords_sorted = list(set(keywords))

    keywords_sorted.sort()

    matcher = PhraseMatcher(nlp.vocab, attr="lower")
    patterns = [nlp.make_doc(text) for text in keywords_sorted]

    matcher.add("KEY_TERMS", patterns, on_match=on_match)

    return matcher


def extract_lexical_features(topic, discipline, output_dir):
    """
    given array of rationales,
    return dict of arrays holding lexical features for each, including:
        - number of keywords
        - number of equations
    """
    feature_type = "lexical"
    _, df = get_topic_data(topic, discipline, output_dir)
    rationales = df[["rationale", "id"]].values

    lexical_features = {}

    if discipline in DALITE_DISCIPLINES:
        matcher = get_matcher(subject=discipline, nlp=nlp)
        try:
            lexical_features["{}_n_keyterms".format(feature_type)] = {
                arg_id: len(
                    set([str(doc[start:end]) for match_id, start, end in matcher(doc)])
                )
                for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
            }

        except TypeError:
            pass

        matcher_prompt = get_matcher(subject=discipline, topic=topic, nlp=nlp)
        lexical_features["{}_n_prompt_terms".format(feature_type)] = {
            arg_id: len(
                set([str(doc[start:end]) for match_id, start, end in matcher_prompt(doc)])
            )
            for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
        }

        lexical_features["{}_n_equations".format(feature_type)] = {
            arg_id: len(
                [
                    token.text
                    for token in doc
                    if token.text == EQUATION_TAG or token.text == EXPRESSION_TAG
                ]
            )
            for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
        }

        lexical_features["{}_n_OOV".format(feature_type)] = {
            arg_id: len([token.text for token in doc if token.text == OOV_TAG])
            for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
        }

    spell = SpellChecker()
    lexical_features["{}_n_spelling_errors".format(feature_type)] = {
        arg_id: len(
            spell.unknown([token.text for token in doc if token.pos not in DROPPED_POS])
        )
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    return lexical_features


def extract_syntactic_features(topic, discipline, output_dir):
    """
    given array of rationales,
    return dict of arrays holding synctatic features for each, including:
        - num_sents
        - num_verbs
        - num_conj
    """
    feature_type = "syntax"
    _, df = get_topic_data(topic, discipline, output_dir)
    rationales = df[["rationale", "id"]].values

    syntactic_features = {}

    # POS_TAGS = [
    #     (ADJ, "ADJ"),
    #     (ADP, "ADP"),
    #     (ADV, "ADV"),
    #     (AUX, "AUX"),
    #     (CONJ, "CONJ"),
    #     (CCONJ, "CCONJ"),
    #     (NOUN, "NOUN"),
    #     (NUM, "NUM"),
    #     (PRON, "PRON"),
    #     (SCONJ, "SCONJ"),
    #     (SYM, "SYM"),
    #     (VERB, "VERB"),
    #     (X, "X"),
    # ]
    # for pos_tag, pos_tag_name in POS_TAGS:
    #     feature_name = "{}_n_{}".format(feature_type, pos_tag_name)
    #
    #     syntactic_features[feature_name] = {
    #         arg_id: len([token for token in doc if token.pos == pos_tag])
    #         for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    #     }

    syntactic_features["{}_n_negations".format(feature_type)] = {
        arg_id: len([token for token in doc if token.dep_ == "neg"])
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    syntactic_features["{}_n_VERB_mod".format(feature_type)] = {
        arg_id: len([token for token in doc if token.tag_ == "MD"])
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    syntactic_features["{}_n_PRON_pers".format(feature_type)] = {
        arg_id: len([token for token in doc if token.tag_ == "PRP"])
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    syntactic_features["{}_dep_tree_depth".format(feature_type)] = {
        arg_id: np.mean([tree_height(sent.root) for sent in doc.sents])
        for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
    }

    return syntactic_features


def extract_readability_features(topic, discipline, output_dir):
    """
    given array of rationales,
    return dict of arrays holding synctatic features for each, including:
        - flesch_kincaid_grade_level
        - flesch_kincaid_reading_ease
        - dale_chall
        - automated_readability_index
        - coleman_liau_index
    """
    feature_type = "readability"
    _, df = get_topic_data(topic, discipline, output_dir)
    rationales = df[["rationale", "id"]].values

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
        readability_features["{}_{}".format(feature_type, f)] = {
            arg_id: round(getattr(doc._, f), 3)
            for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
        }

    return readability_features


def extract_semantic_features(topic, discipline, output_dir):

    feature_type = "semantic"
    semantic_features = {}

    _, df = get_topic_data(topic, discipline, output_dir)
    rationales = df[["rationale", "id"]].values

    if discipline in DALITE_DISCIPLINES:
        df_q = get_questions_df(discipline=discipline)
        texts = df_q.loc[
            df_q["title"].str.startswith(topic), ["text", "expert_rationale"]
        ].values
        q = nlp(texts[0][0])

        semantic_features["{}_sim_question".format(feature_type)] = {
            arg_id: doc.similarity(q)
            for doc, arg_id in nlp.pipe(rationales, batch_size=20, as_tuples=True)
        }

        if type(texts[0][1]) != float:
            expert_rationale = nlp(texts[0][1])
            semantic_features["{}_sim_expert".format(feature_type)] = {
                arg_id: doc.similarity(expert_rationale)
                for doc, arg_id in nlp.pipe(rationales, batch_size=20, as_tuples=True)
            }

    rationales_only_text = [r[0] for r in rationales]
    rationales_only_ids = [r[1] for r in rationales]
    x = np.array(
        [
            doc.vector
            for doc, arg_id in nlp.pipe(rationales, batch_size=100, as_tuples=True)
        ]
    )
    #https://stackoverflow.com/a/41906332
    m, n = x.shape
    distances = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            distances[i, j] = 1 - cosine(x[i, :], x[j, :])
    distances = np.nan_to_num(distances)
    semantic_features["{}_sim_others".format(feature_type)] = {
        arg_id: d
        for arg_id, d in zip(rationales_only_ids, distances.mean(axis=1).round(3))
    }

    return semantic_features


def extract_features_and_save(discipline, topic, feature_type, output_dir):
    """
    dispatch to correct function based on feature type, save result, and return
    feature_dict
    """

    feature_type_fpath = os.path.join(
        output_dir, "data_with_features", feature_type, "{}.json".format(topic),
    )

    print("\t\t\tcalculating features " + feature_type)
    if feature_type == "syntax":
        features = extract_syntactic_features(
            topic=topic, discipline=discipline, output_dir=output_dir
        )
    elif feature_type == "readability":
        features = extract_readability_features(
            topic=topic, discipline=discipline, output_dir=output_dir
        )
    elif feature_type == "lexical":
        features = extract_lexical_features(
            topic=topic, discipline=discipline, output_dir=output_dir
        )
    elif feature_type == "convincingness":
        features = extract_convincingness_features(
            topic=topic, discipline=subject, output_dir=output_dir
        )
    elif feature_type == "surface":
        features = extract_surface_features(
            topic=topic, discipline=discipline, output_dir=output_dir
        )
    elif feature_type == "semantic":
        features = extract_semantic_features(
            topic=topic, discipline=discipline, output_dir=output_dir
        )
    with open(feature_type_fpath, "w") as f:
        json.dump(features, f, indent=2)

    return features


def get_features(topic, discipline, feature_type, output_dir):
    """
    append features onto df_answers
    """

    if feature_type == "all":
        feature_types = ["surface", "lexical", "syntax", "readability", "semantic"]
    else:
        feature_types = [feature_type]

    _, df_answers = get_topic_data(
        topic=topic, discipline=discipline, output_dir=output_dir
    )

    for feature_type in feature_types:
        features = extract_features_and_save(
            topic=topic,
            discipline=discipline,
            feature_type=feature_type,
            output_dir=output_dir,
        )
        for f in features:
            df_answers = pd.merge(
                df_answers,
                pd.DataFrame(features[f], columns=["id", f]),
                on="id",
                how="left",
            )

    return df_answers


def append_features(
    topic, discipline, feature_types_included, output_dir, timestep=None
):
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

    _, df = get_topic_data(topic, discipline, output_dir)

    # append columns with pre-calculated features
    features_dir = os.path.join(output_dir, "data_with_features")
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
            df[feature] = df["id"].map(str).map(features_json[feature])
    return df


def main(
    discipline: (
        "Discipline",
        "positional",
        None,
        str,
        ["Physics", "Chemistry", "Ethics", "same_teacher_two_groups","UKP","IBM_ArgQ","IBM_Evi"],
    ),
    feature_type: (
        "Feature Type",
        "positional",
        None,
        str,
        ["all", "surface", "readability", "lexical", "syntax", "semantic"],
    ),
    output_dir_name: ("Directory name for results", "positional", None, str,),
    largest_first: ("Largest Files First", "flag", "l", bool,),
):
    print("{} - {}".format(discipline, feature_type))
    print("Start: {}".format(datetime.datetime.now()))

    output_dir = os.path.join(
        data_loaders.BASE_DIR, "tmp", output_dir_name, discipline, "all"
    )

    if feature_type == "all":
        feature_types = ["surface", "lexical", "syntax", "readability", "semantic"]
    else:
        feature_types = [feature_type]

    for feature_type in feature_types:
        results_sub_dir = os.path.join(output_dir, "data_with_features", feature_type)

        # make directory if doesn't exist
        pathlib.Path(results_sub_dir).mkdir(parents=True, exist_ok=True)

        data_dir_discipline = os.path.join(output_dir, "data")

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
                topic=topic,
                discipline=discipline,
                feature_type=feature_type,
                output_dir=output_dir,
            )

            df_all = pd.concat([df_all, df_topic])

        fp = os.path.join(
            results_sub_dir, f"all_topics_with_features_{feature_type}.csv"
        )
        df_all.to_csv(fp)
        print("Finished {}: {} ".format(feature_type, datetime.datetime.now()))


if __name__ == "__main__":
    import plac

    plac.call(main)
