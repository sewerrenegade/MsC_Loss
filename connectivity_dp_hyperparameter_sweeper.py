import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import itertools
import hashlib
import numpy as np
import time
import fcntl  # For file locking on Unix-based systems
from .connectivity_dp_experiment import ConnectivityHyperParamExperiment

class ConnectivityDPHyperparameterSweeper:
    def __init__(self, config, name, results_dir="connectivity_downprojection_results", n_repeats=1):
        self.config = config
        self.n_repeats = n_repeats
        self.config_hash = self.hash_dict()
        self.name = name
        self.results_dir = results_dir
        self.folder_path = os.path.join(results_dir, f"{self.name}_{self.n_repeats}r_{self.config_hash}")
        os.makedirs(self.folder_path, exist_ok=True)
        self._save_config_dict()
        self.results_file = os.path.join(self.folder_path, f"{self.name}_results.csv")
        self.progress_file = os.path.join(self.folder_path, f"{self.name}_progress.json")
        self.lock_file = f"{self.progress_file}.lock"
        self.param_combinations = list(itertools.product(*self.config.values()))
        self.columns = list(self.config.keys())
        self.results = self._load_previous_results()

    def _save_config_dict(self):
        with open(f"{self.folder_path}/config.json", "w") as f:
            json.dump(self.config, f, indent=4)

    def hash_dict(self) -> str:
        self.config["nb_repeats"] = self.n_repeats
        sorted_dict = json.dumps(self.config, sort_keys=True)
        hash_object = hashlib.sha256(sorted_dict.encode('utf-8'))
        del self.config["nb_repeats"]
        return hash_object.hexdigest()[:5]

    def _load_previous_results(self):
        if os.path.exists(self.results_file):
            return pd.read_csv(self.results_file).to_dict(orient="records")
        return []

    def _save_results(self, result_entry):
        results_df = pd.DataFrame([result_entry])
        results_df.to_csv(self.results_file, mode="a", header=not os.path.exists(self.results_file), index=False)

    def _load_progress(self):
        if not os.path.exists(self.progress_file):
            return set(), set()

        with open(self.lock_file, "w") as lock_f:  # Open lock file for exclusive locking
            fcntl.flock(lock_f, fcntl.LOCK_EX)  # Acquire exclusive lock
            try:
                with open(self.progress_file, "r") as f:
                    try:
                        progress = json.load(f)
                        return set(progress.get("in_progress", [])), set(progress.get("completed_idx", []))
                    except json.JSONDecodeError:
                        return set(), set()
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)  # Release the lock after reading

    def _save_progress(self, in_progress, completed):
        with open(self.lock_file, "w") as lock_f:  # Open lock file for exclusive locking
            fcntl.flock(lock_f, fcntl.LOCK_EX)  # Acquire exclusive lock
            try:
                with open(self.progress_file, "w") as f:
                    json.dump({"in_progress": list(in_progress), "completed_idx": list(completed)}, f)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)  # Release the lock after writing

    def run_experiment(self, **params):
        exp = ConnectivityHyperParamExperiment(**params)
        return exp.run_experiment()

    def sweep(self):
        in_progress, completed_experiments = self._load_progress()

        while True:
            next_experiment = None
            for i, param_set in enumerate(self.param_combinations):
                if i not in completed_experiments and i not in in_progress:
                    next_experiment = (i, param_set)
                    break

            if next_experiment is None:
                print("All experiments are completed or in progress!")
                break

            i, param_set = next_experiment
            params = dict(zip(self.columns, param_set))
            print(f"Starting experiment {i}")
            
            in_progress.add(i)
            self._save_progress(in_progress, completed_experiments)

            try:
                metric_results = []
                figs = []
                for _ in range(self.n_repeats):
                    result_metrics, fig, loss_curve = self.run_experiment(**params)
                    metric_results.append(result_metrics)
                    figs.append(fig)

                aggregated_results = {}
                params["exp_idx"] = i
                for key in metric_results[0]:
                    values = [res[key] for res in metric_results]
                    aggregated_results[f"{key}_mean"] = np.mean(values)
                    aggregated_results[f"{key}_std"] = np.std(values)

                result_entry = {**params, **aggregated_results}
                self._save_results(result_entry)
                self._save_progress(in_progress - {i}, completed_experiments | {i})

            except Exception as e:
                print(f"Error occurred at iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                in_progress.remove(i)
                self._save_progress(in_progress, completed_experiments)
                break
