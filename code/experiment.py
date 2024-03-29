import os
import plac
from pathlib import Path

import argBT
import make_pairs
import data_loaders
import plots

DALITE_DISCIPLINES = ["Physics","Chemistry","Ethics"]

def main(
    discipline:(
        "Discipline",
        "positional",
        None,
        str,
        ["Physics","Ethics","Chemistry","same_teacher_two_groups","UKP","IBM_ArgQ"],

    ),
    output_dir_name: (
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
    ),
    largest_first: ("Largest Files First", "flag", "largest-first", bool,),
    time_series_validation_flag: ("Time Series Validation", "flag", "time-series", bool,),
):
    if filter_switchers:
        output_dir = os.path.join(data_loaders.BASE_DIR, "tmp", output_dir_name, discipline,"switchers")
    else:
        output_dir = os.path.join(data_loaders.BASE_DIR, "tmp", output_dir_name, discipline,"all")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if discipline in DALITE_DISCIPLINES:
        make_pairs.main(
            discipline=discipline,
            output_dir=output_dir,
            filter_switchers=filter_switchers
        )
        rank_score_types = ["winrate_no_pairs","winrate","elo","crowd_BT","BT"]
    else:
        rank_score_types = ["winrate","elo","BT","reference"]

    for rank_score_type in rank_score_types:
        argBT.main(
            discipline=discipline,
            rank_score_type=rank_score_type,
            output_dir_name=output_dir_name,
            largest_first=largest_first,
            time_series_validation_flag=time_series_validation_flag
        )

    plots.main(
        figures="all",
        discipline=discipline,
        output_dir_name=output_dir_name,
        filter_switchers=filter_switchers,
    )

if __name__ == '__main__':
    plac.call(main)
