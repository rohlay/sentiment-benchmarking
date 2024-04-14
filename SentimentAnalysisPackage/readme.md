

### 1. Data Preprocessing
# Scripts: preprocess.py
>> python preprocess.py dataset.csv
Input: dataset.csv,             1st column sentiment values, 2nd column text, no header
Output: dataset-processed.csv,  1st column sentiment values, 2nd column text, added header 
                                                                              ('sentiment', 'text')

# Scripts: preprocess-check.py (optional)
# Scripts: reduce.py (not finished)

### 2. Sentiment Classifiers
# Scripts: models_evaluation.py 
>> python -m sentiment_analysis_project.scripts.models_evaluation   


### 3. Visualization
# Plots and subplots
# Scripts: plots1.py 
python -m sentiment_analysis_project.scripts.visualization.plots1

# Subplot combine mean and median
# Scripts: plots2.py 
python -m sentiment_analysis_project.scripts.visualization.plots2