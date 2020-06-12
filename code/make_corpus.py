import os
import collections
import pandas as pd
import plac
import data_loaders


# from peerinst.models import Question,Answer

VOTES_MIN = 10
MIN_RECORDS_PER_QUESTION = 50
MIN_WORD_COUNT = 10

def get_shown_rationales(row):
    shown = list(
        ShownRationale.objects.filter(shown_for_answer=row["id"])
        .exclude(shown_answer=row["chosen_rationale_id"])
        .exclude(shown_answer__isnull=True)
        .values_list(
            "shown_answer__id",
            flat=True
        )
    )
    return shown if len(shown)>0 else [0]

def get_mydalite_answers():

    # code below is to be run from directory where dalite-django app is running
    ###
    # q_list=Question.objects.filter(
    #     discipline__title__in=["Physics","Chemistry","Biology","Statistics"]
    # ).values_list("id",flat=True)
    #
    # # remove students who have not given consent
    # usernames_to_exclude = (
    #     Consent.objects.filter(tos__role="student")
    #     .values("user__username")
    #     .annotate(Max("datetime"))
    #     .filter(accepted=False)
    #     .values_list("user",flat=True)
    # )
    #
    # # answers where students chose a peer's explanation over their own
    # answers=Answer.objects.filter(
    #     chosen_rationale_id__isnull=False,
    #     question_id__in=q_list
    # ).exclude(user_token__in=usernames_to_exclude)

    # answers_df= pd.DataFrame(
    #         answers.values(
    #             "id",
    #             "user_token",
    #             "question_id",
    #             "first_answer_choice",
    #             "second_answer_choice",
    #             "rationale",
    #             "chosen_rationale_id",
    #             "datetime_second"
    #         )
    #     ).rename(columns={"datetime_second":"timestamp_rationale"})

    # # rank answers by time of submission
    # answers_df["a_rank_by_time"]=(
    #     answers_df.groupby("question_id")["id"].rank()
    # )
    # # get chosen rationales (not just id)
    # chosen_rationales=Answer.objects.filter(
    #     id__in=answers.values_list(
    #         "chosen_rationale",flat=True
    #     )
    # ).values("id","user_token","rationale","datetime_second")
    #
    # chosen_rationales_df=pd.DataFrame(
    #         chosen_rationales
    #     ).rename(
    #         columns={
    #             "id":"chosen_rationale_id",
    #             "rationale":"chosen_rationale",
    #             "user_token":"chosen_student",
    #             "datetime_second":"timestamp_chosen_rationale"
    #         }
    #     )
    #
    # # add rank by time for chosen rationale
    # df=pd.merge(
    #     answers_df,
    #     chosen_rationales_df,
    #     on="chosen_rationale_id"
    # )
    #
    # # add rank by time for chosen rationale
    # df=pd.merge(
    #     (
    #         df[["id","a_rank_by_time"]]
    #         .rename(columns={
    #             "id":"chosen_rationale_id",
    #             "a_rank_by_time":"chosen_a_rank_by_time",
    #         })
    #     ),
    #     df,
    #     on="chosen_rationale_id",
    #     how="right"
    # )
    # # load data on questions so as to append columns on first/second correct
    # path_to_data=os.path.join(data_loaders.BASE_DIR,"data","2020_03_18__all_questions.csv")
    # all_q = pd.read_csv(path_to_data)
    # df=pd.merge(
    #     df,
    #     all_q.loc[:,["id","correct_answerchoice","discipline","title"]].rename(columns={"id":"question_id"}),
    #     on="question_id"
    # )
    # df["first_correct"]=df["first_answer_choice"]==df["correct_answerchoice"].apply(lambda x: x[1]).map(int)
    # df["second_correct"]=df["second_answer_choice"]==df["correct_answerchoice"].apply(lambda x: x[1]).map(int)
    # df.loc[(df["first_correct"]==True)&(df["second_correct"]==True),"transition"]="rr"
    # df.loc[(df["first_correct"]==True)&(df["second_correct"]==False),"transition"]="rw"
    # df.loc[(df["first_correct"]==False)&(df["second_correct"]==True),"transition"]="wr"
    # df.loc[(df["first_correct"]==False)&(df["second_correct"]==False),"transition"]="ww"

    # df["rationales"]=df.apply(lambda x: get_shown_rationales(x),axis=1)
    # print(df.shape)
    # fpath="/home/sbhatnagar/PhD/convincingness_project/mydalite_answers_{}.csv".format(datetime.datetime.today().strftime("%Y_%m_%d"))
    # df.to_csv(fpath)
    # print(fpath)

    fpath = os.path.join(
        data_loaders.BASE_DIR, os.pardir, "mydalite_answers_2020_06_11.csv"
    )
    df = pd.read_csv(fpath)

    return df


def get_ethics_answers():
    fname = (
        "/home/sbhatnagar/PhD/convincingness_project/data_harvardx/dalite_20161101.csv"
    )

    df = pd.read_csv(fname)

    df = df.rename(columns={"edx_user_hash_id": "user_token", "rationale_id": "id","second_check_time":"timestamp_rationale"})
    df["chosen_rationale_id"] = df["chosen_rationale_id"].astype(int)
    df["topic"] = (
        df["question_text"].str.strip("[?.,]").apply(lambda x: max(x.split(), key=len))
    )
    df["topic"] = df["question_id"].astype(str) + "_" + df["topic"]
    df["discipline"] = "Ethics"
    df["transition"] = (
        (df["first_answer_choice"] == df["second_answer_choice"].astype(int))
        .astype(int)
        .map({0: "switch_ans", 1: "same_ans"})
    )

    df=df[~df["rationale"].isna()]

    return df


def filter_df_answers(df):

    print("all")
    print(df.shape)

    df_switchers = df[df["chosen_rationale_id"] != df["id"]].copy()

    print("all switchers")
    print(df_switchers.shape)

    df2 = df_switchers[(df_switchers["rationale_word_count"] >= MIN_WORD_COUNT)].copy()

    print("\n wc filter")
    print(df2.shape)

    # df2 = df[(
    #     abs(df["chosen_rationale_word_count"]-df["rationale_word_count"])<=MAX_WORD_COUNT_DIFF
    # )].copy()

    # ensure that each explanation has been chosen a minimum number of times for reliability,
    votes = df["chosen_rationale_id"].value_counts()

    df3 = df2[
        ((df2["chosen_rationale_id"].isin(votes[votes >= VOTES_MIN].index)))
    ].copy()

    print("\n vote_min filter")
    print(df3.shape)

    records_per_question = df3["topic"].value_counts()

    df_filtered = df3[
        (
            (
                df3["topic"].isin(
                    records_per_question[
                        records_per_question >= MIN_RECORDS_PER_QUESTION
                    ].index
                )
            )
            #     &(df["user_token"].isin(student_list))
        )
    ].copy()

    print("\n q record filter")
    print(df_filtered.shape)

    return df_filtered


def make_pairs(df):
    """
    Function that takes answer level observations and converts to pairs
    Arguments:
        - df: all answers
    Returns:
        - pandas Dataframe of pairs
    """
    output_dir = os.path.join(
        data_loaders.BASE_DIR, "data", "mydalite_arg_pairs_others"
    )

    df_filtered = filter_df_answers(df)

    df_dalite = pd.DataFrame()
    for topic, df_question in df_filtered.groupby("topic"):

        ranked_pairs = []

        # balanced classes
        for i, (index, row) in enumerate(df_question.iterrows()):
            dr = {}
            if i % 2 == 0:
                dr = {
                    "a1": row["rationale"],
                    "a2": row["chosen_rationale"],
                    "label": "a2",
                    "a1_id": "arg" + str(row["id"]),
                    "a2_id": "arg" + str(row["chosen_rationale_id"]),
                    "a2_author": df[df["id"] == row["chosen_rationale_id"]][
                        "user_token"
                    ].iat[0]
                    if df[df["id"] == row["chosen_rationale_id"]]["user_token"].shape[0]
                    != 0
                    else "",
                    "a1_author": row["user_token"],
                }
            else:
                dr = {
                    "a1": row["chosen_rationale"],
                    "a2": row["rationale"],
                    "label": "a1",
                    "a2_id": "arg" + str(row["id"]),
                    "a1_id": "arg" + str(row["chosen_rationale_id"]),
                    "a1_author": df[df["id"] == row["chosen_rationale_id"]][
                        "user_token"
                    ].iat[0]
                    if df[df["id"] == row["chosen_rationale_id"]]["user_token"].shape[0]
                    != 0
                    else "",
                    "a2_author": row["user_token"],
                }

            dr["#id"] = "{}_{}".format(dr["a1_id"], dr["a2_id"])
            # dr["transition"]=row["transition"]
            dr["annotator"] = row["user_token"]
            ranked_pairs.append(dr)
            dr

            try:
                others_df = pd.DataFrame(
                    [
                        {
                            "shown_answer__id": int(k),
                            "shown_answer__rationale": df.loc[
                                df["id"] == int(k), "rationale"
                            ].iat[0],
                            "shown_answer__user_token": df.loc[
                                df["id"] == int(k), "user_token"
                            ].iat[0],
                        }
                        for k in row["rationales"].strip("[]").split(",")
                        if df.loc[df["id"] == int(k), "rationale"].shape[0] != 0
                    ]
                )
            except AttributeError:
                others_df = pd.DataFrame()


            if others_df.shape[0] > 0:
                # word counts
                others_df["shown_rationale_word_count"] = others_df[
                    "shown_answer__rationale"
                ].str.count("\w+")

                others_df=others_df[others_df["shown_rationale_word_count"]>=MIN_WORD_COUNT]

                # others_df[np.abs(others_df["shown_rationale_word_count"]-row["chosen_rationale_word_count"])<=MAX_WORD_COUNT_DIFF]
                for j, (i2, p) in enumerate(others_df.iterrows()):
                    dr = {}
                    if j % 2 == 0:
                        dr = {
                            "a1": p["shown_answer__rationale"],
                            "a2": row["chosen_rationale"],
                            "label": "a2",
                            "a1_id": "arg" + str(p["shown_answer__id"]),
                            "a2_id": "arg" + str(row["chosen_rationale_id"]),
                            "a1_author": p["shown_answer__user_token"],
                            "a2_author": row["user_token"],
                        }
                    else:
                        dr = {
                            "a1": row["chosen_rationale"],
                            "a2": p["shown_answer__rationale"],
                            "label": "a1",
                            "a2_id": "arg" + str(p["shown_answer__id"]),
                            "a1_id": "arg" + str(row["chosen_rationale_id"]),
                            "a1_author": row["user_token"],
                            "a2_author": p["shown_answer__user_token"],
                        }

                    dr["#id"] = "{}_{}".format(dr["a1_id"], dr["a2_id"])
                    dr["transition"] = row["transition"]
                    dr["annotator"] = row["user_token"]

                    ranked_pairs.append(dr)

        df_rank = pd.DataFrame(ranked_pairs)
        df_rank["y"] = df_rank["label"].map({"a1": -1, "a2": 1})
        df_rank["topic"] = topic

        # # exclude pairs which have an argument that only appears once
        arg_appearance_counts = collections.Counter(
            df_rank["a1_id"].to_list() + df_rank["a2_id"].to_list()
        )
        exclude_args = [k for k, v in arg_appearance_counts.items() if v == 1]
        df_rank = df_rank[
            (~df_rank["a1_id"].isin(exclude_args))
            | (~df_rank["a2_id"].isin(exclude_args))
        ].copy()

        df_dalite = pd.concat([df_dalite, df_rank])

        discipline = df_question["discipline"].value_counts().index[0]

        fname = "{}_{}.csv".format(discipline, topic.replace("/", "_"),)
        fpath = os.path.join(output_dir, fname)
        df_rank.to_csv(fpath)

    return df_dalite


def make_all_pairs():

    df_mydalite = get_mydalite_answers()
    df_mydalite["topic"] = df_mydalite["title"]

    # word counts
    df_mydalite["rationale_word_count"] = df_mydalite["rationale"].str.count("\w+")

    df_pairs_mydalite = make_pairs(df_mydalite)
    print("mydalite pairs : {}".format(df_pairs_mydalite.shape[0]))
    df_ethics = get_ethics_answers()
    # word counts
    df_ethics["rationale_word_count"] = df_ethics["rationale"].str.count("\w+")
    df_ethics["discipline"]="Ethics"

    df_pairs_ethics = make_pairs(df_ethics)
    print("ethics pairs : {}".format(df_pairs_ethics.shape[0]))

    df_pairs_all = pd.concat([df_pairs_mydalite, df_pairs_ethics])

    return df_pairs_all


def main():

    df_all = make_all_pairs()


if __name__ == "__main__":
    import plac

    plac.call(main)


# archive
#   RESULTS_FPATH= "/home/sbhatnagar/PhD/convincingness_project/convincingness/data/"

# hyperparamaters = {
#     "VOTES_MIN":VOTES_MIN,
# #     "STD_FRAC":STD_FRAC,
#     "MAX_WORD_COUNT_DIFF":MAX_WORD_COUNT_DIFF,
#     "MIN_RECORDS_PER_QUESTION":MIN_RECORDS_PER_QUESTION,
#     "N":df_filtered.shape[0],
#     "n_questions":df_filtered["question_id"].value_counts().shape[0],
#     "n_students":df_filtered["user_token"].value_counts().shape[0],
#     "avg_q_per_student":np.round(df_filtered.groupby(["user_token"]).size().mean(),0),
#     "frac_switch":int((
#         df_filtered[
#             (df_filtered["transition"]=="wr")
#             |(df_filtered["transition"]=="rw")
#         ].shape[0]/df_filtered.shape[0])*100)
# }
#
# for var_name,value in hyperparamaters.items():
#     fname=os.path.join(RESULTS_FPATH,var_name+".tex")
#     with open(fname,"w") as f:
#         f.write(str(value))

# table of transitions by discipline
# d=df_filtered.groupby(["discipline"])["transition"].value_counts()
# d.name="N"
# d=d.to_frame().reset_index().pivot(index="discipline",columns="transition",values="N").round(2)
# fname=os.path.join(RESULTS_FPATH,"transitions_by_discipline.tex")
# with open(fname,"w") as f:
#     f.write(d.to_latex(index_names=False))

##
