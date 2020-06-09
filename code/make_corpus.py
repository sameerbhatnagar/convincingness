import os
import pandas as pd
import plac

from django.conf import settings
from peerinst.models import Question,Answer

RESULTS_FPATH= "/home/sbhatnagar/PhD/convincingness_project/convincingness/data/"
VOTES_MIN=5
# STD_FRAC=0.5
MAX_WORD_COUNT_DIFF=25
MIN_RECORDS_PER_QUESTION=50

def main():


    q_list=Question.objects.filter(
        discipline__title__in=["Physics","Chemistry","Biology","Statistics"]
    ).values_list("id",flat=True)
    # answers where students chose a peer's explanation over their own
    answers=Answer.objects.filter(
        chosen_rationale_id__isnull=False,
        question_id__in=q_list
    )
    chosen_rationales=Answer.objects.filter(
        id__in=answers.values_list(
            "chosen_rationale",flat=True
        )
    ).values("id","user_token","rationale","datetime_second")

    df=pd.merge(
        pd.DataFrame(
            answers.values(
                "id",
                "user_token",
                "question_id",
                "first_answer_choice",
                "second_answer_choice",
                "rationale",
                "chosen_rationale_id",
                "datetime_second"
            )
        ).rename(columns={"datetime_second":"timestamp_chosen"}),
        pd.DataFrame(
            chosen_rationales
        ).rename(
            columns={
                "id":"chosen_rationale_id",
                "rationale":"chosen_rationale",
                "user_token":"chosen_student",
                "datetime_second":"timestamp_written"
            }
        ),
        on="chosen_rationale_id"
    )

    # load data on questions so as to append columns on first/second correct
    path_to_data=os.path.join(settings.BASE_DIR,os.pardir,"convincingness","2020_03_18__all_questions.csv")
    all_q = pd.read_csv(path_to_data)
    df=pd.merge(
        df,
        all_q.loc[:,["id","correct_answerchoice","discipline"]].rename(columns={"id":"question_id"}),
        on="question_id"
    )
    df["first_correct"]=df["first_answer_choice"]==df["correct_answerchoice"].apply(lambda x: x[1]).map(int)
    df["second_correct"]=df["second_answer_choice"]==df["correct_answerchoice"].apply(lambda x: x[1]).map(int)
    df.loc[(df["first_correct"]==True)&(df["second_correct"]==True),"transition"]="rr"
    df.loc[(df["first_correct"]==True)&(df["second_correct"]==False),"transition"]="rw"
    df.loc[(df["first_correct"]==False)&(df["second_correct"]==True),"transition"]="wr"
    df.loc[(df["first_correct"]==False)&(df["second_correct"]==False),"transition"]="ww"


    print(df.shape)
    print(pd.DataFrame(
        Question.objects.filter(
            pk__in=df.groupby(["question_id"]).size().index
        ).values("title","discipline__title")
    )["discipline__title"].value_counts())

    # word counts
    df["rationale_word_count"] = df["rationale"].str.count(
        "\w+"
    )
    df["chosen_rationale_word_count"] = df["chosen_rationale"].str.count(
        "\w+"
    )

    df2 = df[(
        abs(df["chosen_rationale_word_count"]-df["rationale_word_count"])<=MAX_WORD_COUNT_DIFF
    )].copy()

    print("\n wc filter")
    print(df2.shape)
    print(pd.DataFrame(
        Question.objects.filter(
            pk__in=df2.groupby(["question_id"]).size().index
        ).values("title","discipline__title")
    )["discipline__title"].value_counts())


    # ensure that each explanation has been chosen a minimum number of times for reliability,
    # but less than a maximum number of times to avoid having too many pairs with the same chosen rationale
    votes=df["chosen_rationale_id"].value_counts()

    df3 = df2[(
        (df2["chosen_rationale_id"].isin(votes[votes>=VOTES_MIN].index))
    )].copy()

    print("\n vote_min filter")
    print(df3.shape)
    print(pd.DataFrame(
        Question.objects.filter(
            pk__in=df3.groupby(["question_id"]).size().index
        ).values("title","discipline__title")
    )["discipline__title"].value_counts())

    records_per_question=df3["question_id"].value_counts()

    df_filtered = df3[(
        (df3["question_id"].isin(records_per_question[records_per_question>=MIN_RECORDS_PER_QUESTION].index))
    #     &(df["user_token"].isin(student_list))
    )].copy()

    print("\n q record filter")
    print(df_filtered.shape)
    pd.DataFrame(
        Question.objects.filter(
            pk__in=df_filtered.groupby(["question_id"]).size().index
        ).values("title","discipline__title")
    )["discipline__title"].value_counts()

    hyperparamaters = {
        "VOTES_MIN":VOTES_MIN,
    #     "STD_FRAC":STD_FRAC,
        "MAX_WORD_COUNT_DIFF":MAX_WORD_COUNT_DIFF,
        "MIN_RECORDS_PER_QUESTION":MIN_RECORDS_PER_QUESTION,
        "N":df_filtered.shape[0],
        "n_questions":df_filtered["question_id"].value_counts().shape[0],
        "n_students":df_filtered["user_token"].value_counts().shape[0],
        "avg_q_per_student":np.round(df_filtered.groupby(["user_token"]).size().mean(),0),
        "frac_switch":int((
            df_filtered[
                (df_filtered["transition"]=="wr")
                |(df_filtered["transition"]=="rw")
            ].shape[0]/df_filtered.shape[0])*100)
    }

    for var_name,value in hyperparamaters.items():
        fname=os.path.join(RESULTS_FPATH,var_name+".tex")
        with open(fname,"w") as f:
            f.write(str(value))

    # table of transitions by discipline
    d=df_filtered.groupby(["discipline"])["transition"].value_counts()
    d.name="N"
    d=d.to_frame().reset_index().pivot(index="discipline",columns="transition",values="N").round(2)
    fname=os.path.join(RESULTS_FPATH,"transitions_by_discipline.tex")
    with open(fname,"w") as f:
        f.write(d.to_latex(index_names=False))

    ##
    output_dir = "/home/sbhatnagar/PhD/convincingness_project/mydalite_arg_pairs_others"

    df_dalite=pd.DataFrame()
    for question_id,df_question in df_filtered.groupby("question_id"):

        ranked_pairs=[]

        # balanced classes
        for i,(index,row) in enumerate(
            df_question[
                [
                    "id",
                    "chosen_rationale_id",
                    "user_token",
                    "rationale",
                    "chosen_rationale",
                    "transition",
                    "chosen_rationale_word_count",
                ]
            ].iterrows()
        ):
            dr={}
            if i%2==0:
                dr={
                    "a1":row["rationale"],
                    "a2":row["chosen_rationale"],
                    "label":"a2",
                    "a1_id": "arg"+str(row["id"]),
                    "a2_id": "arg"+str(row["chosen_rationale_id"]),
                }
            else:
                dr={
                    "a1":row["chosen_rationale"],
                    "a2":row["rationale"],
                    "label":"a1",
                    "a2_id": "arg"+str(row["id"]),
                    "a1_id": "arg"+str(row["chosen_rationale_id"]),
                }

            dr["#id"]= "{}_{}".format(dr["a1_id"],dr["a2_id"])
            dr["transition"]=row["transition"]
            dr["annotator"]=row["user_token"]
            dr["author"]=Answer.objects.get(id=row["id"]).user_token
    #         dr["timestamp"]=Answer.objects.get(id=row["id"]).datetime_second
            ranked_pairs.append(dr)

            # other shown rationales not chosen
            others_df=pd.DataFrame(ShownRationale.objects.filter(
                        shown_for_answer=row["id"]
                    ).exclude(
                        shown_answer=row["chosen_rationale_id"]
                    ).exclude(
                shown_answer__isnull=True
            ).values(
                "shown_answer__rationale",
                "shown_answer__id",
                "shown_answer__user_token",
    #             "shown_answer__datetime_second"
            )
            )

            if others_df.shape[0]>0:
                # word counts
                others_df["shown_rationale_word_count"] = others_df["shown_answer__rationale"].str.count(
                    "\w+"
                )
                others_df[np.abs(others_df["shown_rationale_word_count"]-row["chosen_rationale_word_count"])<=MAX_WORD_COUNT_DIFF]
                for j,(i2,p) in enumerate(others_df.iterrows()):
                    dr={}
                    if j%2==0:
                        dr={
                            "a1":p["shown_answer__rationale"],
                            "a2":row["chosen_rationale"],
                            "label":"a2",
                            "a1_id": "arg"+str(p["shown_answer__id"]),
                            "a2_id": "arg"+str(row["chosen_rationale_id"]),
                        }
                    else:
                        dr={
                            "a1":row["chosen_rationale"],
                            "a2":p["shown_answer__rationale"],
                            "label":"a1",
                            "a2_id": "arg"+str(p["shown_answer__id"]),
                            "a1_id": "arg"+str(row["chosen_rationale_id"]),
                        }


                    dr["#id"]= "{}_{}".format(dr["a1_id"],dr["a2_id"])
                    dr["transition"]=row["transition"]
                    dr["annotator"]=row["user_token"]
                    dr["author"]=p["shown_answer__user_token"]

                    ranked_pairs.append(dr)


        df_rank=pd.DataFrame(ranked_pairs)
        df_rank["y"]=df_rank["label"].map({"a1":-1,"a2":1})
        df_rank["topic"]=question_id
        df_dalite=pd.concat([df_dalite,df_rank])

        fname="{}_{}.csv".format(
            Question.objects.get(pk=question_id).discipline.title,
            Question.objects.get(pk=question_id).title.replace("/","_")
        )
        fpath=os.path.join(output_dir,fname)
        df_rank.to_csv(fpath)


if __name__=="__main__":
    import plac; plac.call(main)
