import datetime

import nltk

from sklearn.feature_extraction.text import CountVectorizer

# nltk.download("punkt")
# nltk.download("wordnet")


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def get_unique_args(df):
    return len(
        list(
            set(
                df["a1_id"].value_counts().index.to_list()
                + df["a2_id"].value_counts().index.to_list()
            )
        )
    )


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def get_vectorizer(term_freq=False, lemmatize=False, idf=True):
    if term_freq:
        if lemmatize:
            return TfidfVectorizer(
                tokenizer=LemmaTokenizer(), use_idf=idf, token_pattern=None
            )
        else:
            return TfidfVectorizer(use_idf=idf)
    else:
        if lemmatize:
            return CountVectorizer(tokenizer=LemmaTokenizer(), token_pattern=None)
        else:
            return CountVectorizer()


def get_corpus(df):
    all_args_train = pd.concat(
        [
            df[["a1_id", "a1"]].rename(columns={"a1_id": "id", "a1": "a"}),
            df[["a2_id", "a2"]].rename(columns={"a2_id": "id", "a2": "a"}),
        ]
    )

    corpus = all_args_train.drop_duplicates("id")["a"]
    return corpus


def get_vocab(corpus):
    vec = get_vectorizer()
    vec.fit(corpus)
    return vec.get_feature_names()
