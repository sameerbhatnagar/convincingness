import os
import pandas as pd
import html as ihtml
from bs4 import BeautifulSoup
import gensim
import spacy

nlp=spacy.load("en_core_web_md")
MIN_SIM_SCORE = 0.75

from data_loaders import BASE_DIR
from utils_scrape_openstax import OPENSTAX_TEXTBOOK_DISCIPLINES

# https://www.kaggle.com/ceshine/remove-html-tags-using-beautifulsoup
def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text)).text
    # text = re.sub(r"http[s]?://\S+", "", text)
    # text = re.sub(r"\s+", " ", text)
    return text


def get_questions_df(discipline):

    if discipline in ["Physics", "Chemistry"]:
        fp = os.path.join(BASE_DIR, os.pardir, "all_questions.csv")
        df_q = pd.read_csv(fp)

    elif discipline == "Ethics":

        data_dir = os.path.join(BASE_DIR, os.pardir,"data_harvardx")

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


def book_corpus_reader(discipline,model_name="doc2vec"):
    """
    Arguments:
    ----------
        discipline -> str
        model_name -> str; optional

    Returns:
    --------
        generator of either:
            - gensim.doc2vec.TaggedDocument
            - if model=="lsi" : lists of tokens for each document
    """
    books=OPENSTAX_TEXTBOOK_DISCIPLINES[discipline]
    fnidx=-1
    for book in books:
        book_dir = os.path.join(
            BASE_DIR, os.pardir, "textbooks", discipline, book
        )
        files = [f for f in os.listdir(book_dir) if not 'key-terms' in f]
        files.sort()
        for fp in files:
            fp=os.path.join(book_dir,fp)
            with open(fp,"r") as f:
                for i,line in enumerate(f):
                    fnidx+=1
                    tokens = [
                        token.text.lower() for token in nlp(line)
                        if not token.is_punct and token.is_alpha
                    ]
                    if model_name=="lsi":
                        yield " ".join(tokens)
                    else:
                        f_id=f"{os.path.basename(fp)}_{i}_{fnidx}"
                        yield gensim.models.doc2vec.TaggedDocument(tokens,[f_id])


def build_lsi_model(discipline):
    """
    Arguments:
    ----------
        discipline -> str

    Returns:
    --------
        model : trained gensim.LsiModel
        dictionary : gensim.Dictionary
        corpus : list of BoW represenetations
    """

    documents=list(book_corpus_reader(discipline,model_name="lsi"))
    texts=[
        [
            word for word in document.lower().split()
            if word not in nlp.Defaults.stop_words
            and w2c.get(word,0)>1
        ]
        for document in documents
    ]
    dictionary=corpora.Dictionary(texts)
    corpus=[dictionary.doc2bow(text) for text in texts]
    lsi_model=models.LsiModel(corpus,id2word=dictionary,num_topics=100)

    return lsi_model, dictionary, corpus, documents

def get_reference_texts(topic,discipline,dictionary,lsi_model,index,documents):
    """
    Arguments:
    ----------
    - topic (question title) -> str
    - discipline -> str
    - dictionary -> loaded gensim.corpora.Dictionary
    - lsi_model -> gensim.models.LsiModel
    - index -> gensim.similarities.docsim.MatrixSimilarity
    - documents -> list of lists of tokens

        e.g. lsi_model, dictionary, corpus, documents = build_lsi_model(discipline)


    Returns:
    --------
    - similar_reference_texts -> list of reference passages from OpenStax text that are deemed similar to
        topic prompt, using Lsi model
    """
    df_q = get_questions_df(discipline=discipline)
    q=df_q[df_q["title"]==topic].text.iat[0]

    q_tokens=[token.text for token in nlp(q) if token.is_alpha and not token.is_punct]

    vec_bow=dictionary.doc2bow(q_tokens)
    vec_lsi=lsi_model[vec_bow]
    sims=index[vec_lsi]

    sims=index[vec_lsi]
    sims=sorted(enumerate(sims),key=lambda item: -item[1])

    similar_reference_texts = []
    for doc_position, doc_score in sims:
        if doc_score >= MIN_SIM_SCORE:
            similar_reference_texts.append(documents[doc_position])

    # image_alt_text = df_q.loc[()
    #     (~df_q["image_alt_text"].isna())
    #     &(df_q["title"]==topic)
    # ),"image_alt_text"].iat[0]
    # if image_alt_text:
    #     image_alt_text_tokens=[token.text for token in nlp(image_alt_text) if token.is_alpha and not token.is_punct]

    return similar_reference_texts
