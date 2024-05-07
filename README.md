
# Execution Steps
**1.	Setting up the Python environment: Run pip install -r requirements.txt to install all required packages.**
**2.	Initialization**
      - Download word embedding files and place them in /input_data/glove_embeddings.
      - Download lexicon files and place them in /lexicon_files/.
      - Insert the dataset data.csv into /datasets/raw. After preprocessing, relocate data-processed.csv to /datasets/processed.
**3. Evaluation Process**
   - For machine learning analysis:
     - 3.1.	Adjust config_ml.json as necessary.
     - 3.2.	Execute models_evaluation.py.
     - 3.3.	A folder will appear in /experiment_outputs/exp_ID1
   - For lexicon-based analysis:
     - 3.4.	Adjust config_lex.json accordingly.
     - 3.5.	Execute lexicon_evaluation_norm.py.
**4.	Post-processing Steps**
    - To standardize machine learning regression metrics, place the metrics folder in
   /post_processing/machine_learning_data/metrics_ml and run normalize_continuous_metrics.py.
**5.	Visualization Execution**
    - For machine learning visualizations:
      - Ensure the metrics are located in /post_processing/machine_learning_data/metrics_ml.
      - Execute either ML_plots_and_subplots.py or ML_subplots_violin_mean_median.py.
    - For lexicon visualizations:
      - Ensure the metrics are positioned in /post_processing/lexicon_data/metrics_lex.
      - Run LEX_plots_and_subplots.py.
