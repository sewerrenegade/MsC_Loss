import pandas as pd
import os
import json
import itertools
import hashlib
import json

from configs.global_config import GlobalConfig
from models.topology_models.custom_topo_tools.connectivity_dp_experiment import ConnectivityHyperParamExperiment

class ConnectivityDPHyperparameterSweeper:
    def __init__(self, config, name):
        """
        Initializes the hyperparameter sweeper.

        Args:
            config (dict): Dictionary containing parameter names and their respective values.
            name (str): Unique name for the sweeper, used for saving results.
        """
        self.config = config
        self.config_hash = self.hash_dict()
        self.name = name
        self.folder_path = GlobalConfig.CONNECTIVITY_DP_SWEEPER_PATH + f"{self.name}_{self.config_hash}/"
        os.makedirs(self.folder_path, exist_ok=True)
        self._save_config_dict()
        self.results_file = f"{self.folder_path}{self.name}_results.csv"
        self.progress_file = f"{self.folder_path}{self.name}_progress.json"
        self.param_combinations = list(itertools.product(*self.config.values()))
        self.columns = list(self.config.keys())
        self.results = self._load_previous_results()

    def _save_config_dict(self):
        with open(f"{self.folder_path}config.json", "w") as f:
            json.dump(self.config,f,indent=4)
 
    def hash_dict(self) -> str:
        """
        Hashes a dictionary in a consistent way, regardless of key order.
        Assumes the dictionary contains only JSON-serializable primitives.
        """
        sorted_dict = json.dumps(self.config, sort_keys=True)
        hash_object = hashlib.sha256(sorted_dict.encode('utf-8'))
        return hash_object.hexdigest()[:5]
    
    def _load_previous_results(self):
        """
        Loads previous results and progress if they exist.

        Returns:
            list: List of previously completed results.
        """
        if os.path.exists(self.results_file):
            print(f"Loading previous results from {self.results_file}...")
            return pd.read_csv(self.results_file).to_dict(orient="records")
        return []

    def _save_results(self):
        """
        Saves the current results to a CSV file.
        """
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.results_file, index=True)
        print(f"Results saved to {self.results_file}")

    def _save_progress(self, completed_idx):
        """
        Saves the progress to a JSON file.

        Args:
            completed_idx (int): Index of the last completed experiment.
        """
        with open(self.progress_file, "w") as f:
            json.dump({"completed_idx": completed_idx}, f)
        print(f"Progress saved to {self.progress_file}")

    def _load_progress(self):
        """
        Loads the progress from a JSON file.

        Returns:
            int: Index of the last completed experiment, or -1 if no progress file exists.
        """
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                progress = json.load(f)
            return progress.get("completed_idx", -1)
        return -1

    def run_experiment(self, **params):
        """
        Mock experiment function. Replace this with your actual experiment logic.

        Args:
            **params: Parameter values for the experiment.

        Returns:
            dict: Experiment results.
        """
        print(f"Running connectivity down projection with the following parameters:{params}")
        exp = ConnectivityHyperParamExperiment(**params)
        return exp.run_experiment()
    
    def _save_experiment_figure(self,fig,index):
        viz_path = f"{self.folder_path}vizualizations/{index}.png"
        fig.savefig(viz_path, format=format, bbox_inches="tight")
        print(f"Plot saved as {viz_path}")
    def sweep(self):
        """
        Executes the hyperparameter sweep and saves the results after every iteration.
        """
        # Determine the starting point
        start_idx = self._load_progress() + 1

        for i, param_set in enumerate(self.param_combinations[start_idx:], start=start_idx):
            try:
                # Unpack parameters
                params = dict(zip(self.columns, param_set))

                # Run the experiment
                result_metrics, fig = self.run_experiment(**params)

                # Combine input and output for storage
                result_entry = {**params, **result_metrics}
                self.results.append(result_entry)

                # Save results and progress
                self._save_results()
                self._save_experiment_figure(fig,i)
                self._save_progress(i)

            except Exception as e:
                print(f"Error occurred at iteration {i}: {e}")
                break