import os

# Define the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Define the base directory (i.e., project_folder) by going up directories until you reach 'SentimentAnalysisPackage'
BASE_DIR = SCRIPT_DIR
while os.path.basename(BASE_DIR) != 'SentimentAnalysisPackage':
    BASE_DIR = os.path.dirname(BASE_DIR)

# Input directories
DATASETS_DIR = os.path.join(BASE_DIR, 'sentiment_analysis_project', 'input_data', 'datasets')
DATA_PROCESSED = os.path.join(DATASETS_DIR, 'processed')

GLOVE_EMBBEDINGS_PATH = os.path.join(BASE_DIR, 'sentiment_analysis_project', 'input_data', 'glove-embeddings')
GLOVE_ENGLISH_FILE_PATH = os.path.join(GLOVE_EMBBEDINGS_PATH, 'glove.6B.100d.txt')
GLOVE_SPANISH_FILE_PATH = os.path.join(GLOVE_EMBBEDINGS_PATH, 'SBW-vectors-300-min5.txt')

# Lexicon Files
LEXICON_FILES_DIR = os.path.join(BASE_DIR, 'sentiment_analysis_project', 'input_data', 'lexicon_files')
#GENERAL_INQUIRER_FILE = os.path.join(LEXICON_FILES_DIR, 'GI', 'inquirerbasic.xls')
GENERAL_INQUIRER_FILE = os.path.join(LEXICON_FILES_DIR, 'GI', 'inquireraugmented.xls')
MPQA_FILE = os.path.join(LEXICON_FILES_DIR, 'MPQA', 'MPQA.tff')
OPINIONFINDER_FILE = os.path.join(LEXICON_FILES_DIR, 'Opinion_Finder', 'subjcluesSentenceClassifiersOpinionFinderJune06.tff')


# JSON files
CONFIG_ML = os.path.join(BASE_DIR, 'sentiment_analysis_project', 'scripts', 'ml_config.json')
CONFIG_LEXICON = os.path.join(BASE_DIR, 'sentiment_analysis_project', 'scripts', 'lex_config.json')

# Output directories
OUTPUTS_DIR = os.path.join(BASE_DIR, 'sentiment_analysis_project', 'experiments_outputs')

# Postprocessing Directories
POST_PROCESSING = os.path.join(BASE_DIR, 'sentiment_analysis_project', 'scripts', 'post_processing')
POST_PROC_TIMES = os.path.join(POST_PROCESSING, 'times')

# Machine Learning
POST_PROC_ML = os.path.join(POST_PROCESSING, 'machine_learning_data')
METRICS_ML = os.path.join(POST_PROC_ML, 'metrics_ml')
PLOTS_ML = os.path.join(POST_PROC_ML, 'plots_ml')

# Lexicon
POST_PROC_LEX = os.path.join(POST_PROCESSING, 'lexicon_data')
METRICS_LEX = os.path.join(POST_PROC_LEX, 'metrics_lex')
PLOTS_LEX = os.path.join(POST_PROC_LEX, 'plots_lex')


