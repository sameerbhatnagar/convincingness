# Experiments and code

## Experiment 1

What factors predict whether a student will change their explanation on review step?

Are `convincingness` features useful for this prediction task, or are `surface` features enough?

1. Extract data from django database, using commented code in `make_pairs.get_mydalite_answers`. Answers for HarvardX data is in a csv that can be loaded using `make_pairs.get_ethics_answers`.

2. `$python make_pairs.py <discipline>`
 - this will seperate and save `answer` files in a directory `/tmp/switch_exp/<discipline>/data`, and then create pairs (and save) `/tmp/switch_exp/<discipline>/data_pairs`.

3. `$python switch_exp.py <discipline>`
