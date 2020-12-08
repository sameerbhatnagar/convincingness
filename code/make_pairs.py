import os
import json
import collections
import pandas as pd
import plac
import data_loaders
import pathlib
import spacy
import re

nlp = spacy.load("en_core_web_md", disable=["ner"])
EQUATION_TAG = "EQUATION"
EXPRESSION_TAG = "EXPRESSION"
OOV_TAG = "OOV_TAG"

# from peerinst.models import Question,Answer

MIN_RECORDS_PER_QUESTION = 100

def make_pairs_IBM_Evi(output_dir_name):
    """
    function to make per topic files for the IBM_EviConv dataset
    """
    df_ibm_evi=data_loaders.load_arg_pairs_IBM_Evi(train_test_split=False)
    discipline="IBM_Evi"
    population = "all"
    output_dir = os.path.join(data_loaders.BASE_DIR, "tmp", output_dir_name, discipline,population)
    data_dir_discipline=os.path.join(output_dir,"data_pairs")
    for topic,df_topic in df_ibm_evi.groupby("topic"):
        fp=os.path.join(data_dir_discipline,f"{topic}.csv")
        df_topic.to_csv(fp,sep="\t")
    return

def get_shown_rationales(row):
    """
    this function is used in extraction of mydalite data from django db
    """
    shown = list(
        ShownRationale.objects.filter(shown_for_answer=row["id"])
        .exclude(shown_answer=row["chosen_rationale_id"])
        .exclude(shown_answer__isnull=True)
        .values_list("shown_answer__id", flat=True)
    )
    return shown if len(shown) > 0 else [0]


def get_mydalite_answers():

    # code below is to be run from directory where dalite-django app is running
    ###
    # q_list=Question.objects.filter(
    #     discipline__title__in=["Physics","Chemistry","Biology","Statistics"]
    # ).values_list("id",flat=True)
    #
    # remove students who have not given consent
    # usernames_to_exclude = (
    #     Consent.objects.filter(tos__role="student")
    #     .values("user__username")
    #     .annotate(Max("datetime"))
    #     .filter(accepted=False)
    #     .values_list("user",flat=True)
    # )
    # usernames_tos = (
    #     Consent.objects.filter(tos__role="student")
    #     .values("user__username")
    #     .annotate(Max("datetime"))
    #     .filter(accepted=True)
    #     .values_list("user",flat=True)
    # )
    #
    # # answers where students chose a peer's explanation over their own
    #
    # answers=Answer.objects.filter(
    # #     chosen_rationale_id__isnull=False,
    #     question_id__in=q_list,
    # ).exclude(
    #     user_token__in=usernames_to_exclude
    # ).exclude(
    #     first_answer_choice=0
    # )
    # #.filter(
    # #     user_token__in=usernames_tos
    # # )
    # print("answers where students consent and belong to disciplines of interest")
    # print(answers.count())
    #
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
    #
    # print("as df")
    # print(answers_df.shape)
    #
    # # to stay consistent with HarvardX data, stick to my own rationale means chosen_rationale_id = id, not ""
    # answers_df.loc[answers_df["chosen_rationale_id"].isna(),"chosen_rationale_id"]=(
    #     answers_df.loc[answers_df["chosen_rationale_id"].isna(),"id"]
    # )
    # # rank answers by time of submission
    # answers_df["a_rank_by_time"]=(
    #     answers_df.groupby("question_id")["id"].rank()
    # )
    #
    # # get chosen rationales (not just id)
    # chosen_answer_ids = answers_df["chosen_rationale_id"].value_counts().index.to_list()
    # chosen_rationales_df = (
    #     answers_df.loc[answers_df["id"].isin(chosen_answer_ids),
    #                    ["id","user_token","rationale","timestamp_rationale"]
    #                   ]
    # ).rename(
    #         columns={
    #             "id":"chosen_rationale_id",
    #             "rationale":"chosen_rationale",
    #             "user_token":"chosen_student",
    #             "timestamp_rationale":"timestamp_chosen_rationale"
    #         }
    #     )
    #
    #
    # df=pd.merge(
    #     answers_df,
    #     chosen_rationales_df,
    #     on="chosen_rationale_id",
    # )
    #
    # print("after chosen rationale merge")
    # print(df.shape)
    #
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
    #
    # print("after rank by time for chosen rationale")
    # print(df.shape)
    #
    # # load data on questions so as to append columns on first/second correct
    # path_to_data=os.path.join("/home/sbhatnagar/PhD/convincingness_project/convincingness/data/mydalite_metadata/2020_06_09__all_questions.csv")
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
    #
    # df["rationales"]=df.apply(lambda x: get_shown_rationales(x),axis=1)
    # print("final")
    # print(df.shape)
    # fpath="/home/sbhatnagar/PhD/convincingness_project/mydalite_answers_{}.csv".format(datetime.datetime.today().strftime("%Y_%m_%d"))
    # df.to_csv(fpath)
    # print(fpath)
    fpath = os.path.join(
        data_loaders.BASE_DIR, os.pardir, "mydalite_answers_2020_09_18.csv"
    )
    df = pd.read_csv(fpath)

    df["topic"] = df["title"]

    return df

def get_ethics_answers():
    fname = (
        "/home/sbhatnagar/PhD/convincingness_project/data_harvardx/dalite_20161101.csv"
    )

    df = pd.read_csv(fname)

    df = df.rename(
        columns={
            "edx_user_hash_id": "user_token",
            "rationale_id": "id",
            "second_check_time": "timestamp_rationale",
        }
    )
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
    df["a_rank_by_time"] = df.groupby("question_id")["id"].rank()

    df = pd.merge(
        (
            df[["id", "a_rank_by_time"]].rename(
                columns={
                    "id": "chosen_rationale_id",
                    "a_rank_by_time": "chosen_a_rank_by_time",
                }
            )
        ),
        df,
        on="chosen_rationale_id",
        how="right",
    )

    df = df[~df["rationale"].isna()]

    return df


def filter_out_stick_to_own(df):
    df_switchers = df[df["chosen_rationale_id"] != df["id"]].copy()

    # print("all switchers")
    # print(df_switchers.shape)

    return df_switchers


def filter_df_answers(df):

    print("all answers")
    print(df.shape[0])

    records_per_question = df["topic"].value_counts()

    df_filtered = df[
        (
            (
                df["topic"].isin(
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


def clean_rationales(df):
    """
    swap out all equations, expressions, and Out-Of-Vocabulary tokens with
    special placeholder tokens
    """
    eqn_re = re.compile(r"([\w\/^\*\.\(\)+-]+\s?[=]\s?[\w\/^\*\.\(\)+-]+)")
    expr_re = re.compile(r"([\w\/^\*\.\(\)+-]+\s?[+\*\-/]\s?[\w\/^\*\.\(\)+-]+)")

    df["rationale"] = (
        df["rationale"]
        .fillna(" ")
        .str.replace(eqn_re, EQUATION_TAG)
        .str.replace(expr_re, EXPRESSION_TAG)
    )

    df["rationale"] = [
        " ".join([OOV_TAG if token.is_oov == True else token.text for token in doc])
        for doc in nlp.pipe(df["rationale"], batch_size=50)
    ]
    return df


def make_pairs_by_topic(topic, df_unfiltered,filter_switchers=True):
    """
    Arguments:
    =========
        topic : question title
        df_unfiltered
    Returns:
    ========
        df_rank : dataframe with pairs of arguments/rationales, labelled
                    which of the pair was chosen, who the chooser was, and
                    who wrote the chosen rationale (user_token)
    """
    print(topic)
    print("\t{} answers".format(df_unfiltered.shape[0]))
    # pairs are made only from records where students changed their answer choice.
    # the BradleyTerry scores derived are more valid
    # TO DO: show this clearly! Predict winning
    # argument in each pair with derived BT score
    if filter_switchers:
        df_question = filter_out_stick_to_own(df_unfiltered)
        print("\t{} answers where students switched explanations".format(df_question.shape[0]))
    else:
        df_question = df_unfiltered

    ranked_pairs = []

    # balanced classes
    for i, (index, row) in enumerate(df_question.iterrows()):
        # make pairs with answers where student chose someone else's explanation
        if row["id"] != row["chosen_rationale_id"]:
            dr = {}
            if i % 2 == 0:
                dr = {
                    "a1": row["rationale"],
                    "a2": row["chosen_rationale"],
                    "label": "a2",
                    "a1_id": "arg" + str(row["id"]),
                    "a2_id": "arg" + str(row["chosen_rationale_id"]),
                    "a2_author": df_unfiltered[
                        df_unfiltered["id"] == row["chosen_rationale_id"]
                    ]["user_token"].iat[0]
                    if df_unfiltered[df_unfiltered["id"] == row["chosen_rationale_id"]][
                        "user_token"
                    ].shape[0]
                    != 0
                    else "",
                    "a1_author": row["user_token"],
                    "a1_rank_by_time": row["a_rank_by_time"],
                    "a2_rank_by_time": row["chosen_a_rank_by_time"],
                }
            else:
                dr = {
                    "a1": row["chosen_rationale"],
                    "a2": row["rationale"],
                    "label": "a1",
                    "a2_id": "arg" + str(row["id"]),
                    "a1_id": "arg" + str(row["chosen_rationale_id"]),
                    "a1_author": df_unfiltered[
                        df_unfiltered["id"] == row["chosen_rationale_id"]
                    ]["user_token"].iat[0]
                    if df_unfiltered[df_unfiltered["id"] == row["chosen_rationale_id"]][
                        "user_token"
                    ].shape[0]
                    != 0
                    else "",
                    "a2_author": row["user_token"],
                    "a2_rank_by_time": row["a_rank_by_time"],
                    "a1_rank_by_time": row["chosen_a_rank_by_time"],
                }

            dr["#id"] = "{}_{}".format(dr["a1_id"], dr["a2_id"])
            dr["transition"] = row["transition"]
            dr["switch_exp"] = 1
            dr["annotator"] = row["user_token"]
            dr["annotation_rank_by_time"] = row["a_rank_by_time"]
            ranked_pairs.append(dr)

        # if we have data on what was shown, make dataframe of answers
        # that were shown
        try:
            shown_ids = row["rationales"].strip("[]").split(",")
            shown_ids = [k for k in shown_ids if k != ""]
            others_df = pd.DataFrame(
                [
                    {
                        "shown_answer__id": int(k),
                        "shown_answer__rationale": df_unfiltered.loc[
                            df_unfiltered["id"] == int(k), "rationale"
                        ].iat[0],
                        "shown_answer__user_token": df_unfiltered.loc[
                            df_unfiltered["id"] == int(k), "user_token"
                        ].iat[0],
                        "shown_answer__a_rank_by_time": df_unfiltered.loc[
                            df_unfiltered["id"] == int(k), "a_rank_by_time"
                        ].iat[0],
                    }
                    for k in shown_ids
                    if df_unfiltered.loc[
                        df_unfiltered["id"] == int(k), "rationale"
                    ].shape[0]
                    != 0
                ]
            )
        except AttributeError:
            others_df = pd.DataFrame()

        if others_df.shape[0] > 0:
            # word counts
            others_df["shown_rationale_word_count"] = others_df[
                "shown_answer__rationale"
            ].str.count("\w+")


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
                        "a1_rank_by_time": p["shown_answer__a_rank_by_time"],
                        "a2_rank_by_time": row["chosen_a_rank_by_time"],
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
                        "a2_rank_by_time": p["shown_answer__a_rank_by_time"],
                        "a1_rank_by_time": row["chosen_a_rank_by_time"],
                    }

                dr["#id"] = "{}_{}".format(dr["a1_id"], dr["a2_id"])
                dr["transition"] = row["transition"]
                dr["annotator"] = row["user_token"]
                dr["annotation_rank_by_time"] = row["a_rank_by_time"]
                if row["id"] == row["chosen_rationale_id"]:
                    dr["switch_exp"] = 0
                else:
                    dr["switch_exp"] = 1
                ranked_pairs.append(dr)

    df_rank = pd.DataFrame(ranked_pairs)
    # df_rank["y"] = df_rank["label"].map({"a1": -1, "a2": 1})
    df_rank["topic"] = topic

    # # exclude pairs which have an argument that only appears once
    # arg_appearance_counts = collections.Counter(
    #     df_rank["a1_id"].to_list() + df_rank["a2_id"].to_list()
    # )
    # exclude_args = [k for k, v in arg_appearance_counts.items() if v == 1]
    # df_rank = df_rank[
    #     (~df_rank["a1_id"].isin(exclude_args)) | (~df_rank["a2_id"].isin(exclude_args))
    # ].copy()
    print("\t{} pairs".format(df_rank.shape[0]))
    return df_rank



def label_switch_exp(df):
    # label students with a 1 if they chose a peer explanation on review step, otherwise 0
    df.loc[
        (df["chosen_rationale_id"] != df["id"]),
        "switch_exp",
    ] = 1
    df.loc[df["switch_exp"].isna(), "switch_exp"] = 0

    return df


def make_all_pairs(data_file_dict, output_dir,filter_switchers):
    """
    Function that takes answer level observations and converts to pairs
    Arguments:
        - df_answers_all: all answers
        - output_dir: where to save pairs
        - filter_switchers: bool
    Returns:
        - pandas Dataframe of pairs
    """

    all_topics = [os.path.basename(fp) for fp in data_file_dict.values()]

    topics_already_done = [
        "_".join(os.path.basename(t).split("_")[1:])
        for t in os.listdir(os.path.join(output_dir, "data_pairs"))
    ]

    files_to_do = [
        os.path.join(output_dir,"data",t) for t in all_topics
        if t not in topics_already_done
    ]

    df_pairs_all = pd.DataFrame()
    for fp in files_to_do:
        df_topic = pd.read_csv(fp)
        topic=os.path.basename(fp).replace(".csv","")
        df_pairs = make_pairs_by_topic(topic=topic, df_unfiltered=df_topic,filter_switchers=filter_switchers)

        # save
        data_dir = os.path.join(output_dir, "data_pairs")

        fp = os.path.join(data_dir, "pairs_{}.csv".format(topic.replace("/", "_")))
        df_pairs.to_csv(fp)

        df_pairs_all = pd.concat([df_pairs_all,df_pairs])

    print("all pairs made")

    return df_pairs_all


def main(
    discipline:(
        "Discipline",
        "positional",
        None,
        str,
        ["Physics","Ethics","Chemistry","same_teacher_two_groups"],

    ),
    output_dir: (
        "Directory name for results",
        "positional",
        None,
        str,
    ),
    filter_switchers: (
        "keep only students who switch their answer",
        "flag",
        "switch",
        bool,
    )
):

    data_dir = os.path.join(output_dir,"data")
    pathlib.Path(data_dir).mkdir(parents=True,exist_ok=True)
    pathlib.Path(os.path.join(output_dir,"data_pairs")).mkdir(parents=True,exist_ok=True)

    data_file_dict = {}
    if discipline == "Ethics":
        df_answers_all_unfiltered = get_ethics_answers()
        df_answers_all_unfiltered["discipline"] = "Ethics"

    elif discipline == "same_teacher_two_groups":
        df=get_mydalite_answers()
        df["date"]=pd.to_datetime(df["timestamp_rationale"])
        fp="/home/sbhatnagar/PhD/convincingness_project/group_student_lists_phys102.json"
        with open(fp,"r") as f:
            groups=json.load(f)

        # answers for studnets in these groups
        df_groups=pd.DataFrame()
        for g in groups:
            df_g=df[
                (df["user_token"].isin(g["students"]))&(df["date"].dt.month<5)
            ].copy()
            df_g["group"]=g["name"]
            df_groups=pd.concat([df_groups,df_g])

        all_students_groups=list(set(groups[0]["students"]+groups[1]["students"]))
        q_list=df_groups["question_id"].value_counts()[df_groups["question_id"].value_counts()>500].index.tolist()
        df_answers_other_students=df[
            (~df["user_token"].isin(all_students_groups))&(df["question_id"].isin(q_list))
        ]
        df_answers_other_students["group"]="other"
        df_answers_all_unfiltered=pd.concat([df_groups,df_answers_other_students])

        # filter question list again
        q_list=df_answers_all_unfiltered["question_id"].value_counts()[df_answers_all_unfiltered["question_id"].value_counts()>500].index.tolist()
        df_answers_all_unfiltered = df_answers_all_unfiltered[df_answers_all_unfiltered["question_id"].isin(q_list)].copy()

    else:
        df_answers_all_unfiltered_all_disciplines = get_mydalite_answers()
        df_answers_all_unfiltered = df_answers_all_unfiltered_all_disciplines[
            df_answers_all_unfiltered_all_disciplines["discipline"] == discipline
        ]
        # free up memory
        del df_answers_all_unfiltered_all_disciplines

    # filter out based on at least MIN_RECORDS_PER_QUESTION
    df_answers_all = filter_df_answers(df_answers_all_unfiltered)

    # free up memory
    del df_answers_all_unfiltered

    df_answers_all = label_switch_exp(df_answers_all)

    for topic,df_topic in df_answers_all.groupby("topic"):
        fp=os.path.join(
            data_dir,
            "{}.csv".format(topic.replace("/", "_"))
        )
        df_topic = clean_rationales(df_topic.copy())

        df_topic.to_csv(fp)
        data_file_dict[topic]=fp

    # free up memory
    del df_answers_all

    df_pairs_all = make_all_pairs(data_file_dict=data_file_dict, output_dir=output_dir,filter_switchers=filter_switchers)
    print("{} pairs : {}".format(discipline,df_pairs_all.shape[0]))


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
