import os, json, plac
from collections import Counter
import pandas as pd
import numpy as np
from argBT import (
    get_topic_data,
    get_rankings_wc,
    get_rankings_winrate_no_pairs,
    get_rankings_winrate,
    get_rankings_BT,
    get_rankings_elo,
    get_rankings_crowdBT,
)
import data_loaders
from feature_extraction import append_features, nlp,DROPPED_POS
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# https://stackoverflow.com/a/48949667
# https://stackoverflow.com/a/60234758
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator

import matplotlib.pyplot as plt

MIN_WORD_COUNT = 10
MIN_TIMES_SHOWN = 3

class SMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        #         X=X.todense()
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()

    def predict(self, X):
        # check that fit has been called
        check_is_fitted(self, "model_")

        # input validation
        X = check_array(X)

        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)

    def get_params(self, deep=False):
        return {"fit_intercept": self.fit_intercept, "model_class": self.model_class}

    def summary(self):
        print(self.results_.summary())


# https://stackoverflow.com/a/28384887
class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def load_data(discipline, output_dir_name, feature_types_included):
    """
    load data and append features
    """
    # load data and append features
    population = "all"

    data_dir_discpline = os.path.join(
        data_loaders.BASE_DIR, "tmp", output_dir_name, discipline, population, "data"
    )
    output_dir = os.path.join(data_dir_discpline, os.pardir)
    topics = os.listdir(data_dir_discpline)
    df = pd.DataFrame()
    print("1) loading data")
    for t, topic in enumerate(topics):
        if t % 10 == 0:
            print(f"{t}/{len(topics)}")
        df_topic_with_features = append_features(
            topic.replace(".csv", ""), discipline, feature_types_included, output_dir
        )

        pairs_df, _ = get_topic_data(
            topic=topic.replace(".csv", ""),
            discipline=discipline,
            output_dir=output_dir,
        )
        df_topic_with_features["arg_id"] = "arg" + df_topic_with_features["id"].astype(
            str
        )

        _, args_dict = get_rankings_winrate(pairs_df)
        df_topic_with_features["y_winrate"] = df_topic_with_features["arg_id"].map(
            args_dict
        )

        _, args_dict, _ = get_rankings_crowdBT(pairs_df)
        df_topic_with_features["y_crowdBT"] = df_topic_with_features["arg_id"].map(
            args_dict
        )

        _, args_dict = get_rankings_elo(pairs_df)
        df_topic_with_features["y_elo"] = df_topic_with_features["arg_id"].map(
            args_dict
        )

        _, args_dict = get_rankings_BT(pairs_df)
        df_topic_with_features["y_BT"] = df_topic_with_features["arg_id"].map(args_dict)

        _, args_dict = get_rankings_winrate_no_pairs(df_topic_with_features)
        df_topic_with_features["y_winrate_nopairs"] = df_topic_with_features[
            "arg_id"
        ].map(args_dict)

        df = pd.concat([df, df_topic_with_features])

    return df


def normalize_and_rename_columns(df,feature_types_included):
    """
    normalize features by word count, change column names
    """
    print("2) normalize and get col sets")
    df2 = df.copy()
    feature_cols_to_normalize = [
        "surface_n_content_words",
        "lexical_n_keyterms",
        "lexical_n_prompt_terms",
        "lexical_n_equations",
        "lexical_n_spelling_errors",
        "syntax_n_negations",
        "syntax_n_VERB_mod",
        "syntax_n_PRON_pers",
    ]
    normalized_feature_names = []
    for c in feature_cols_to_normalize:
        normalized_feature_name = c.replace("_n_", "_")
        df2[normalized_feature_name] = df2[c] / df2["surface_n_words"]
        df2 = df2.drop(c, axis=1)
        normalized_feature_names.append(normalized_feature_name)
    df2["surface_timerank"] = df2["a_rank_by_time"]
    all_cols = normalized_feature_names + [
        "surface_timerank",
        "surface_TTR",
        "surface_TTR_content",
        "surface_n_sents",
        "readability_flesch_kincaid_grade_level",
        "readability_flesch_kincaid_reading_ease",
        "readability_dale_chall",
        "readability_automated_readability_index",
        "readability_coleman_liau_index",
        "syntax_dep_tree_depth",
        "semantic_sim_question",
        "semantic_sim_others",
    ]
    all_cols.sort()

    # append one-hot-encoded columns for transition type
    drop_tr_col = df["transition"].value_counts().index.tolist()[-1]

    transition_cols = [
        t for t in df["transition"].value_counts().index.tolist() if t != drop_tr_col
    ]

    df2 = pd.concat(
        [
            df2.drop("transition", axis=1),
            pd.get_dummies(df2["transition"]).drop(drop_tr_col, axis=1),
        ],
        axis=1,
    )
    # define which feature sets go with with which columns
    baseline_cols = ["id", "surface_n_words"] + transition_cols
    cols_sets = {
        "baseline": baseline_cols,
        "all_features": all_cols,
        "all": baseline_cols + all_cols,
    }

    for feature_type in feature_types_included:
        cols_sets[feature_type] = [
            c for c in all_cols if c.split("_")[0] == feature_type
        ]
    return df2, cols_sets


def get_bin_edges(df_topic):
    min_wc=df_topic["surface_n_words"].min()
    q1=df_topic["surface_n_words"].describe()["25%"]
    q2=df_topic["surface_n_words"].describe()["50%"]
    q3=df_topic["surface_n_words"].describe()["75%"]
    max_wc=df_topic["surface_n_words"].max()
    bin_edges=[min_wc,q1,q2,q3,max_wc]
    if len(set(bin_edges))==len(bin_edges):
        return bin_edges
    else:
        return None

def get_results(cols_sets, targets, df):
    """
    cross validated results over all topics/targets/feature-sets
    """
    print("3) get results")

    topics = df["topic"].value_counts().index.to_list()

    results=[]
    for col_type,cols in cols_sets.items():
        print(col_type)
        for target in targets:
            print(f"\t{target}")
            skipped=[]
            for t,topic in enumerate(topics):
                if t%40==0:
                    print(f"\t\t{t}/{len(topics)}; {len(set(skipped))} topics")

                df_topic_=df[df["topic"]==topic].copy()

                # count how many times each rationale was shown
                times_shown_counter = Counter()
                s = (
                    df_topic_["rationales"]
                    .dropna()
                    .apply(
                        lambda x: [
                            int(k) for k in x.strip("[]").replace(" ", "").split(",") if k != ""
                        ]
                    )
                )
                _ = s.apply(lambda x: times_shown_counter.update(x))
                df_topic_["times_shown"]=df_topic_["id"].map(times_shown_counter)

                # filter out those not shown often enough
                df_topic = df_topic_[df_topic_["times_shown"]>=MIN_TIMES_SHOWN].copy()

                min_wc=df_topic["surface_n_words"].min()
                q1=df_topic["surface_n_words"].describe()["25%"]
                q2=df_topic["surface_n_words"].describe()["50%"]
                q3=df_topic["surface_n_words"].describe()["75%"]
                max_wc=df_topic["surface_n_words"].max()

                bin_edges=get_bin_edges(df_topic)
                if bin_edges:
                    df_topic["wc_bin"]=pd.cut(
                        df_topic["surface_n_words"],
                        bins=bin_edges,
                        labels=["Q1","Q2","Q3","Q4"],
                        include_lowest=True,
                    )

                    # only continue if each subset of data has min 20 records,
                    # since we will be doing 5 fold CV
                    if all(df_topic["wc_bin"].value_counts()>20):
                        for quartile, df_topic_wc_quartile in df_topic.groupby("wc_bin"):
                            kfcv = KFold(n_splits=5)
                            for fold,(train_index, test_index) in enumerate(kfcv.split(df_topic_wc_quartile)):
                                X_train = df_topic_wc_quartile.iloc[train_index]
                                #                 y_train=StandardScaler().fit_transform(
                                #                     X=X_train[target].rank().values.reshape(-1,1)
                                #                 ).reshape(1,-1)[0]
                                y_train = df_topic_wc_quartile.iloc[train_index][target]
                                pipe = Pipeline(
                                    steps=[
                                        ("regress",LinearRegression())
                                    ]
                                )

                                X_test = df_topic_wc_quartile.iloc[test_index]
                                #                 y_true=StandardScaler().fit_transform(
                                #                     X=X_test[target].rank().values.reshape(-1,1)
                                #                 ).reshape(1,-1)[0]

                                if X_test.shape[0]>10:

                                    pipe.fit(X=X_train[cols],y=y_train)

                                    y_test = df_topic_wc_quartile.iloc[test_index][target]
                                    y_pred=pipe.predict(X_test[cols])

                                    # precision at Kr for this fold: Kr is ratio (10%,25%,50%)
                                    for Kr in [10,25,50]:

                                        K = int(Kr/100*X_test.shape[0])
                                        # repeat for top Kr and bottom Kr
                                        for top in [True,False]:
                                            d={}
                                            if top:
                                                d["rank_type"]="top"
                                                true=X_test["id"].iloc[np.argsort(y_test)].tail(K)
                                                pred=X_test["id"].iloc[np.argsort(y_pred)].tail(K)
                                                not_true=X_test.loc[~X_test["id"].isin(true),"id"]
                                                not_pred=X_test.loc[~X_test["id"].isin(pred),"id"]

                                            else:
                                                d["rank_type"]="bottom"
                                                true=X_test["id"].iloc[np.argsort(y_test)].head(K)
                                                pred=X_test["id"].iloc[np.argsort(y_pred)].head(K)
                                                not_true=X_test.loc[~X_test["id"].isin(true),"id"]
                                                not_pred=X_test.loc[~X_test["id"].isin(pred),"id"]

                                            tp=len(set(true)&set(pred))
                                            fp=len(set(not_true)&set(pred))
                                            precision = tp/(tp+fp)

                                            d["K"]=Kr
                                            d["N"]=X_test.shape[0]
                                            d["prec"]=precision
                                            d["Q"]=quartile
                                            d["fold"]=fold
                                            d["topic"]=topic
                                            d["target"]=target
                                            d["model"]=col_type
                                            results.append(d)
                                else:
                                    skipped.append(topic)
    return results


def main(
    discipline: (
        "Discipline",
        "positional",
        None,
        str,
        ["Physics", "Chemistry", "Ethics"],
    ),
    output_dir_name: ("Directory name for results", "positional", None, str,),
    reload_data: ("Reload data", "flag", "reload", bool)
):
    """
    Evaluate performance on the regression task of predicting aggregated
    convincingness score (e.g. crowdBT, winrate, etc.) using different feature
    sets (e.g. syntax, readability, etc).
    Performance is evaluated not by r2 or rmse, but a recall@N aproach,
    where we ask if the predicted scores yield a ranking that captures
    the top N or bottom N ranked explanations
    """
    population = "all"

    output_dir_discpline = os.path.join(
        data_loaders.BASE_DIR, "tmp", output_dir_name, discipline, population
    )

    feature_types_included = ["surface", "lexical", "readability", "syntax", "semantic"]

    fp=os.path.join(output_dir_discpline,f"df_with_POS_{discipline}.csv")
    if reload_data:
        # 1) load data and append features
        df_ = load_data(discipline, output_dir_name,feature_types_included)

        # 2) POS and content words for rationales as new columns
        df_["rationale_pos"]=pd.Series([
            " ".join([
                token.pos_ for token in doc
                if token.pos not in DROPPED_POS
            ]) for doc in nlp.pipe(df_["rationale"],batch_size=50)]
        )
        df_["rationale_content_words"]=pd.Series(
            [
                " ".join([
                    token.lemma_ for token in doc
                    if not token.is_stop
                ])
                for doc in nlp.pipe(df_["rationale"],batch_size=50)]
        )
        df_.to_csv(fp)
        # df_ngram=pd.read_csv(fp)

    else:
        df_=pd.read_csv(fp)

    # 1a) filter out all rationales < MIN_WORD_COUNT
    df=df_[df_["surface_n_words"]>=MIN_WORD_COUNT].copy()


    # 2) normalize features by word count, change column names
    df2, cols_sets = normalize_and_rename_columns(df,feature_types_included)

    # 3) copy relevant data and make pipeline
    targets = ["y_BT", "y_winrate_nopairs", "y_elo", "y_winrate", "y_crowdBT"]

    df3 = df2[["topic","rationales"] + cols_sets["all"] + targets].dropna()

    # 4) cross validated results over all topics/targets/feature-sets
    results = get_results(cols_sets, targets, df3)
    fp=os.path.join(output_dir_discpline,f"results_topN_{discipline}.json")
    print(fp)
    with open(fp,"w") as f:
        json.dump(results,f)

    # # 5) plot results
    # models = [
    #     "baseline",
    #     "surface",
    #     "lexical",
    #     "readability",
    #     "syntax",
    #     "semantic",
    #     "all_features",
    #     "all",
    # ]
    #
    # plt.style.use("ggplot")
    # fig, axs = plt.subplots(len(models), 2, sharex=True, sharey=True, figsize=(12, 24))
    #
    # for m, model in enumerate(models):
    #     for r, rank_type in enumerate(["top", "bottom"]):
    #         ax = axs[m, r]
    #         for target in targets:
    #             df_res[
    #                 (
    #                     (df_res["rank_type"] == rank_type)
    #                     & (df_res["model"] == model)
    #                     & (df_res["target"] == target)
    #                 )
    #             ].groupby("N")["prec"].mean().plot(ax=ax, label=target)
    #             ax.legend()
    #             ax.set_title(f"{model} - {rank_type}")
    #
    # fig.tight_layout()

    return


if __name__ == "__main__":
    plac.call(main)
