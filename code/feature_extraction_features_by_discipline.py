NUMERIC_COLUMNS = {}
NUMERIC_COLUMNS["Physics"]=[
    'surface_TTR', 'surface_TTR_content',
    "readability_flesch_kincaid_grade_level",
    'readability_flesch_kincaid_reading_ease',
    'readability_dale_chall','readability_automated_readability_index',
    'lexical_n_keyterms', 'lexical_n_prompt_terms', 'lexical_n_equations',
    'lexical_n_OOV', 'lexical_n_spelling_errors',
    'syntax_n_negations','syntax_n_VERB_mod', 'syntax_n_PRON_pers',
    'syntax_dep_tree_depth',
    'semantic_sim_question_glove', 'semantic_sim_others_glove',
    "semantic_dist_ref_Lsi_mean","semantic_dist_ref_Lsi_max","semantic_dist_ref_Lsi_mean",
    "semantic_dist_ref_Doc2Vec_mean","semantic_dist_ref_Doc2Vec_max","semantic_dist_ref_Doc2Vec_min",
    # "id"
    ]
NUMERIC_COLUMNS["Chemistry"]=NUMERIC_COLUMNS["Physics"]
NUMERIC_COLUMNS["Ethics"] = NUMERIC_COLUMNS["Physics"]
NUMERIC_COLUMNS["UKP"] = [
    'surface_TTR', 'surface_TTR_content',
    "readability_flesch_kincaid_grade_level",
    'readability_flesch_kincaid_reading_ease',
    'readability_dale_chall','readability_automated_readability_index',
    'lexical_n_OOV', 'lexical_n_spelling_errors',
    'syntax_n_negations','syntax_n_VERB_mod', 'syntax_n_PRON_pers',
    'syntax_dep_tree_depth',
    'semantic_sim_others',
    ]
NUMERIC_COLUMNS["IBM_ArgQ"]=NUMERIC_COLUMNS["UKP"]
NUMERIC_COLUMNS["IBM_Evi"]=NUMERIC_COLUMNS["UKP"]

BINARY_COLUMNS={}
BINARY_COLUMNS["Physics"]=['first_correct','second_correct','switch_exp']
BINARY_COLUMNS["Chemistry"]=BINARY_COLUMNS["Physics"]
BINARY_COLUMNS["Ethics"]=["switch_exp"]
BINARY_COLUMNS["UKP"] = []
BINARY_COLUMNS["IBM_ArgQ"]=BINARY_COLUMNS["UKP"]
BINARY_COLUMNS["IBM_Evi"]=BINARY_COLUMNS["UKP"]

TARGETS = {}
TARGETS["Physics"]=['y_winrate', 'y_elo', 'y_BT', 'y_winrate_nopairs','y_crowdBT']
TARGETS["Chemistry"]=TARGETS["Physics"]
TARGETS["Ethics"]=TARGETS["Physics"]
TARGETS["UKP"] = ['y_reference','y_winrate', 'y_elo', 'y_BT']
TARGETS["IBM_ArgQ"] = TARGETS["UKP"]
TARGETS["IBM_Evi"] = TARGETS["UKP"]
