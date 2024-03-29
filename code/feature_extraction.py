from __future__ import unicode_literals
import os
import json
import re
import spacy
import plac
import pickle
import datetime
import data_loaders
import pathlib
import numpy as np
import gensim

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

from data_loaders import DALITE_DISCIPLINES, get_topic_data, BASE_DIR

from make_pairs import EQUATION_TAG, EXPRESSION_TAG, OOV_TAG

from feature_extraction_reference_texts import (
    clean_text,
    get_questions_df,
    build_similarity_models,
    get_reference_texts,
)

nlp = spacy.load("en_core_web_md", disable=["ner"])

from spellchecker import SpellChecker

DROPPED_POS = [PUNCT, SPACE, X]


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


def get_matcher(subject, topic=None, on_match=None):
    """
    given a subject and nlp object,
    return a phrase mather object that has patterns from
    OpenStax textbook of that discipline
    """

    df_q = get_questions_df(discipline=subject)

    if topic:
        if subject in ["Physics","Chemistry","Biology"]:
            texts = df_q.loc[
                df_q["title"].str.startswith(topic),
                ["text", "image_alt_text", "expert_rationale"],
            ].values
        else:
            texts = df_q.loc[
                df_q["title"].str.startswith(topic),
                ["text", "expert_rationale"],
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
                        if not token.is_stop and token.is_alpha
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

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in keywords_sorted]

    matcher.add("KEY_TERMS", patterns)

    return matcher


def get_bin_edges(df_topic):
    min_wc = df_topic["surface_n_words"].min()
    q1 = df_topic["surface_n_words"].describe()["25%"]
    q2 = df_topic["surface_n_words"].describe()["50%"]
    q3 = df_topic["surface_n_words"].describe()["75%"]
    max_wc = df_topic["surface_n_words"].max()
    bin_edges = [min_wc - 1, q1, q2, q3, max_wc + 1]
    if len(set(bin_edges)) == len(bin_edges):
        return bin_edges
    else:
        return None


def append_wc_quartile_column(df_topic):
    """
    calculate quartiles for word count for given topic
    """
    min_wc = df_topic["surface_n_words"].min()
    q1 = df_topic["surface_n_words"].describe()["25%"]
    q2 = df_topic["surface_n_words"].describe()["50%"]
    q3 = df_topic["surface_n_words"].describe()["75%"]
    max_wc = df_topic["surface_n_words"].max()

    bin_edges = get_bin_edges(df_topic)
    if bin_edges:
        df_topic["wc_bin"] = pd.cut(
            df_topic["surface_n_words"],
            bins=bin_edges,
            labels=["Q1", "Q2", "Q3", "Q4"],
            # include_lowest=True,
        )
    return df_topic


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
        matcher = get_matcher(subject=discipline)
        try:
            lexical_features["{}_n_keyterms".format(feature_type)] = {
                arg_id: len(
                    set([str(doc[start:end]) for match_id, start, end in matcher(doc)])
                )
                for doc, arg_id in nlp.pipe(rationales, batch_size=50, as_tuples=True)
            }

        except TypeError:
            pass

        matcher_prompt = get_matcher(subject=discipline, topic=topic)
        lexical_features["{}_n_prompt_terms".format(feature_type)] = {
            arg_id: len(
                set(
                    [
                        str(doc[start:end])
                        for match_id, start, end in matcher_prompt(doc)
                    ]
                )
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
        arg_id: len([token.text for token in doc if token.is_oov])
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


def filter_for_glove_oov(r):
    return " ".join([token.text for token in nlp(r) if token.has_vector])


def extract_semantic_features(topic, discipline, output_dir, models_dict_reload=False):

    feature_type = "semantic"
    semantic_features = {}

    _, df = get_topic_data(topic, discipline, output_dir)
    df["rationale"] = df["rationale"].apply(filter_for_glove_oov)
    rationales = df[["rationale", "id"]].values

    # build LSI and Doc2Vec models
    if models_dict_reload:
        models_dict = build_similarity_models(discipline)
    else:
        fp = os.path.join(output_dir, "models_dict.pkl")
        with open(fp, "rb") as f:
            models_dict = pickle.load(f)

    reference_texts = get_reference_texts(topic, discipline, models_dict)

    if discipline in DALITE_DISCIPLINES:
        df_q = get_questions_df(discipline=discipline)
        if discipline in ["Physics","Chemistry","Biology"]:
            df_q["text_all"] = df_q[["text", "expert_rationale", "image_alt_text"]].apply(
                lambda x: f"{x['text']}. {x['expert_rationale']}. {x['image_alt_text']}",
                axis=1,
            )
        else:
            df_q["text_all"] = df_q[["text", "expert_rationale"]].apply(
                lambda x: f"{x['text']}. {x['expert_rationale']}.",
                axis=1,
            )
        q_text = df_q[df_q["title"] == topic]["text_all"].iat[0]
        q_text_filtered = " ".join(
            [
                token.text
                for token in nlp(q_text)
                if token.has_vector and not token.is_stop and token.is_alpha
            ]
        )
        q = nlp(q_text_filtered)

        semantic_features["{}_sim_question_glove".format(feature_type)] = {
            arg_id: doc.similarity(q)
            for doc, arg_id in nlp.pipe(rationales, batch_size=20, as_tuples=True)
        }

        # Sim to reference_texts LSI
        model_key = "Lsi"
        model = models_dict[model_key]["model"]
        dictionary = models_dict[model_key]["dictionary"]
        ref_texts = [
            [token.text for token in doc if not token.is_stop]
            for doc in nlp.pipe(reference_texts[model_key])
        ]
        corpus_reference = [dictionary.doc2bow(text) for text in ref_texts]
        index = gensim.similarities.MatrixSimilarity(model[corpus_reference])

        semantic_features[f"{feature_type}_dist_ref_{model_key}_mean"] = {}
        semantic_features[f"{feature_type}_dist_ref_{model_key}_max"] = {}
        semantic_features[f"{feature_type}_dist_ref_{model_key}_min"] = {}
        for a, a_id in rationales:
            tokens = [token.text for token in nlp(a) if not token.is_stop]
            vec_bow = dictionary.doc2bow(tokens)
            vec_lsi = model[vec_bow]
            sims = index[vec_lsi].astype("float64")
            semantic_features[f"{feature_type}_dist_ref_{model_key}_mean"][
                a_id
            ] = sims.mean()
            semantic_features[f"{feature_type}_dist_ref_{model_key}_max"][a_id] = sims.max()
            semantic_features[f"{feature_type}_dist_ref_{model_key}_min"][a_id] = sims.min()

        # sim to ref_texts Doc2Vec
        model_key = "Doc2Vec"
        model = models_dict[model_key]["model"]
        semantic_features[f"{feature_type}_dist_ref_{model_key}_mean"] = {}
        semantic_features[f"{feature_type}_dist_ref_{model_key}_max"] = {}
        semantic_features[f"{feature_type}_dist_ref_{model_key}_min"] = {}
        for a, a_id in rationales:
            tokens = [token.text for token in nlp(a) if not token.is_stop]
            inferred_vector = model.infer_vector(tokens)
            sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
            ref_text_ids = [d["name"] for d in reference_texts[model_key]]
            sim_values = [s[1] for s in sims if s[0] in ref_text_ids]
            semantic_features[f"{feature_type}_dist_ref_{model_key}_mean"][a_id] = np.mean(
                sim_values
            )
            semantic_features[f"{feature_type}_dist_ref_{model_key}_max"][a_id] = np.max(
                sim_values
            )
            semantic_features[f"{feature_type}_dist_ref_{model_key}_min"][a_id] = np.min(
                sim_values
            )

    # sim to others GloVe
    rationales_only_text = [r[0] for r in rationales]
    rationales_only_ids = [r[1] for r in rationales]
    x = np.array(
        [
            doc.vector
            for doc, arg_id in nlp.pipe(rationales, batch_size=100, as_tuples=True)
        ]
    )
    # https://stackoverflow.com/a/41906332
    m, n = x.shape
    distances = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            distances[i, j] = 1 - cosine(x[i, :], x[j, :])
    distances = np.nan_to_num(distances)
    semantic_features["{}_sim_others_glove".format(feature_type)] = {
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
        [
            "Physics",
            "Chemistry",
            "Ethics",
            "same_teacher_two_groups",
            "UKP",
            "IBM_ArgQ",
            "IBM_Evi",
        ],
    ),
    feature_type: (
        "Feature Type",
        "positional",
        None,
        str,
        ["all", "surface", "readability", "lexical", "syntax", "semantic"],
    ),
    population: ("Which students?", "positional", None, str, ["all", "switchers"],),
    output_dir_name: ("Directory name for results", "positional", None, str,),
    largest_first: ("Largest Files First", "flag", "l", bool,),
    filter_topics: (
        "Only work on topics which pass filter criteria in JEDM paper",
        "flag",
        "ft",
        bool,
    ),
):
    print("{} - {}".format(discipline, feature_type))
    print("Start: {}".format(datetime.datetime.now()))

    output_dir = os.path.join(
        data_loaders.BASE_DIR, "tmp", output_dir_name, discipline, population
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

        if filter_topics:
            fp = os.path.join(
                BASE_DIR, "tmp", output_dir_name, discipline, population, "topics.json"
            )
            with open(fp, "r") as f:
                topics_filtered = json.load(f)
            topics = [t for t in topics if t in topics_filtered]

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
