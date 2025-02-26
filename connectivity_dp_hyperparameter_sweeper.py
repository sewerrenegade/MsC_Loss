import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import itertools
import hashlib
import numpy as np

from .connectivity_dp_experiment import ConnectivityHyperParamExperiment

class ConnectivityDPHyperparameterSweeper:
    def __init__(self, config, name,save_dir = "connectivity_downprojection_results", n_repeats=1):
        """
        Initializes the hyperparameter sweeper.

        Args:
            config (dict): Dictionary containing parameter names and their respective values.
            name (str): Unique name for the sweeper, used for saving results.
            n_repeats (int): Number of times to repeat each permutation.
        """
        self.config = config
        self.n_repeats = n_repeats
        self.config_hash = self.hash_dict()
        self.name = name
        self.folder_path = save_dir
        os.makedirs(self.folder_path, exist_ok=True)
        self._save_config_dict()
        self.results_file = f"{self.folder_path}{self.name}_results.csv"
        self.progress_file = f"{self.folder_path}{self.name}_progress.json"
        self.param_combinations = list(itertools.product(*self.config.values()))
        self.columns = list(self.config.keys())
        self.results = self._load_previous_results()

    def _save_config_dict(self):
        with open(f"{self.folder_path}config.json", "w") as f:
            json.dump(self.config, f, indent=4)

    def hash_dict(self) -> str:
        """Hashes a dictionary consistently, ignoring key order."""
        self.config["nb_repeats"] = self.n_repeats
        sorted_dict = json.dumps(self.config, sort_keys=True)
        hash_object = hashlib.sha256(sorted_dict.encode('utf-8'))
        del self.config["nb_repeats"]
        return hash_object.hexdigest()[:5]
    
    def _load_previous_results(self):
        if os.path.exists(self.results_file):
            print(f"Loading previous results from {self.results_file}...")
            return pd.read_csv(self.results_file).to_dict(orient="records")
        return []

    def _save_results(self):
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.results_file, index=False)
        print(f"Results saved to {self.results_file}")

    def _save_progress(self, completed_idx):
        with open(self.progress_file, "w") as f:
            json.dump({"completed_idx": completed_idx}, f)
        print(f"Progress saved to {self.progress_file}")

    def _load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                progress = json.load(f)
            return progress.get("completed_idx", -1)
        return -1

    def run_experiment(self, **params):
        """Runs the experiment and returns results."""
        print(f"Running connectivity down projection with parameters: {params}")
        exp = ConnectivityHyperParamExperiment(**params)
        return exp.run_experiment()
    
    def _save_loss_curve(self,loss_curve, index):
        loss_curve_folder_path = f"{self.folder_path}loss_curves/"
        os.makedirs(loss_curve_folder_path, exist_ok=True)
        loss_curve_path = os.path.join(loss_curve_folder_path,f"{index}.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, len(loss_curve) + 1), loss_curve, marker='o', linestyle='-')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss Curve of exp {index}")
        ax.grid(True)
        fig.savefig(loss_curve_path, format='png', bbox_inches="tight")
        plt.close(fig)
        
    def _save_experiment_figure(self, fig, index):
        viz_folder_path = f"{self.folder_path}visualizations/"
        os.makedirs(viz_folder_path, exist_ok=True)
        viz_path = f"{viz_folder_path}{index}.png"
        fig.savefig(viz_path, format='png', bbox_inches="tight")
        plt.close(fig)

    def sweep(self):
        """Executes the hyperparameter sweep and saves results after each iteration."""
        start_idx = self._load_progress() + 1

        for i, param_set in enumerate(self.param_combinations[start_idx:], start=start_idx):
            try:
                params = dict(zip(self.columns, param_set))
                
                # Store multiple runs for each parameter setting
                metric_results = []
                figs = []
                for _ in range(self.n_repeats):
                    result_metrics, fig, loss_curve = self.run_experiment(**params)
                    metric_results.append(result_metrics)
                    figs.append(fig)
                
                # Compute mean and std for each metric
                aggregated_results = {}
                params["exp_idx"] = i
                for key in metric_results[0]:  # Assume all runs return same keys
                    values = [res[key] for res in metric_results]
                    aggregated_results[f"{key}_mean"] = np.mean(values)
                    aggregated_results[f"{key}_std"] = np.std(values)

                result_entry = {**params, **aggregated_results}
                self.results.append(result_entry)
                
                # Save results and progress
                self._save_results()
                self._save_experiment_figure(figs[0], i)  # Save one sample figure
                self._save_loss_curve(loss_curve,i)
                self._save_progress(i)
            
            except Exception as e:
                print(f"Error occurred at iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                break


