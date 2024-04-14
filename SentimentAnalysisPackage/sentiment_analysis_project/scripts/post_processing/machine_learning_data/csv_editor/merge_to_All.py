"""
In case models_evaluation.py is interrupted.
To merge csv files from modules into all
"""

from sentiment_analysis_project.scripts.config import CONFIG_ML
def merge_results():
        # Create 'All' directory
        all_dir_path = f"{self.results_dir}/All"
        os.makedirs(all_dir_path, exist_ok=True)

        # For each problem type and statistic type, merge the results and save into the 'All' directory
        problem_types = ['classification', 'regression'] if self.has_classification and self.has_regression else ['classification'] if self.has_classification else ['regression']
        statistic_types = ['mean', 'median', 'max', 'min']
        
        """
        with open(CONFIG_ML, 'r') as f:
            config = json.load(f)
        module_keys = [module_name for module_name, module_info in config['modules'].items() if module_info['active']]
        """

        module_keys = ['Machine Learning', 'Deep Learning', 'Transformers']


        for problem_type in problem_types:
            for statistic_type in statistic_types:
                for metric_name in self.metric_stats[module_keys[0]].keys():  # Use the first module key to get the metric names
                    # Create an empty DataFrame for each metric
                    final_df = pd.DataFrame()
                    for module_key in module_keys:
                        metric_data = self.metric_stats[module_key][metric_name]
                        for dataset_name, dataset_data in metric_data.items():
                            for model_name, model_data in dataset_data.items():
                                # Retrieve statistic
                                statistic_value = model_data.get(statistic_type, None)
                                if statistic_value is not None:
                                    # Add statistic to DataFrame
                                    final_df.at[dataset_name, f"{model_name}"] = statistic_value

                    # If the DataFrame is not empty, save it as a CSV file
                    if not final_df.empty:
                        dst_file_path = os.path.join(all_dir_path, f"{problem_type}/{statistic_type}/{metric_name}.csv")
                        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                        final_df.to_csv(dst_file_path)
                    else:
                        print(f"No data to merge for problem type '{problem_type}', statistic type '{statistic_type}', and metric '{metric_name}'")
