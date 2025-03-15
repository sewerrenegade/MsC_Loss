import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import itertools
import hashlib
import numpy as np
import time
import random
import portalocker  # Cross-platform file locking
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
        self.experiment_run_order = list(range(len(self.param_combinations)))
        shuffle_seed = os.getpid() + int(time.time())
        rng_state = random.getstate()
        random.seed(shuffle_seed)
        random.shuffle(self.experiment_run_order)
        random.setstate(rng_state)

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

    def _save_results_safe(self, result_entry):
        results_df = pd.DataFrame([result_entry])
        with open(self.results_file, "a+", newline='') as f:  # Prevent Python from adding extra newlines
            portalocker.lock(f, portalocker.LOCK_EX)  # Lock file to prevent race conditions
            file_empty = (f.tell() == 0)  # Check if file is empty (to write headers)
            results_df.to_csv(f, header=file_empty, index=False)  # Use single newline
            portalocker.unlock(f)  # Release lock


    def _load_progress(self):
        if not os.path.exists(self.progress_file):
            return set(), set()
        with open(self.progress_file, "r") as f:
            portalocker.lock(f, portalocker.LOCK_EX)  # Lock file for reading
            try:
                progress = json.load(f)
                return set(progress.get("in_progress", [])), set(progress.get("completed_idx", []))
            except json.JSONDecodeError:
                return set(), set()
            finally:
                portalocker.unlock(f)

    def _save_progress(self, in_progress, completed):
        with open(self.progress_file, "w") as f:
            portalocker.lock(f, portalocker.LOCK_EX)  # Lock file before writing
            json.dump({"in_progress": list(in_progress), "completed_idx": list(completed)}, f)
            portalocker.unlock(f)  # Release lock

    def run_experiment(self, **params):
        exp = ConnectivityHyperParamExperiment(**params)
        return exp.run_experiment()

    def sweep(self):
        print(f"Starting sweep of config: {self.name}")
        while True:
            next_experiment = None
            in_progress, completed_experiments = self._load_progress()
            
            for i in self.experiment_run_order:
                if i not in completed_experiments and i not in in_progress:
                    next_experiment = (i, self.param_combinations[i])
                    in_progress.add(i)
                    self._save_progress(in_progress, completed_experiments)
                    break
            
            
            if next_experiment is None:
                print("All experiments are completed or in progress!")
                break

            i, param_set = next_experiment
            params = dict(zip(self.columns, param_set))
            print(f"Starting experiment {i}")

            try:
                metric_results = []
                for run_nb in range(self.n_repeats):
                    result_metrics, fig, loss_curve = self.run_experiment(**params)
                    if run_nb != self.n_repeats - 1:
                        plt.close(fig)
                        loss_curve
                    metric_results.append(result_metrics)


                aggregated_results = {f"{key}_mean": np.mean([res[key] for res in metric_results]) for key in metric_results[0]}
                aggregated_results.update({f"{key}_std": np.std([res[key] for res in metric_results]) for key in metric_results[0]})
                result_entry = {**params, **aggregated_results}
                result_entry["experiment_index"] = i

                self._save_experiment_figure(fig, i)
                self._save_loss_curve(loss_curve, i)
                self._save_results_safe(result_entry)

                in_progress, completed_experiments = self._load_progress()
                completed_experiments.add(i)
                in_progress.discard(i)
                self._save_progress(in_progress, completed_experiments)

            except Exception as e:
                print(f"Error occurred at iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                in_progress, completed_experiments = self._load_progress()
                in_progress.discard(i)
                self._save_progress(in_progress, completed_experiments)
                break

    def _save_loss_curve(self, loss_curve, index):
        loss_curve_folder_path = os.path.join(self.folder_path, "loss_curves")
        os.makedirs(loss_curve_folder_path, exist_ok=True)
        loss_curve_path = os.path.join(loss_curve_folder_path, f"{index}.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, len(loss_curve) + 1), loss_curve, marker='o', linestyle='-')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss Curve of exp {index}")
        ax.grid(True)
        fig.savefig(loss_curve_path, format='png', bbox_inches="tight")
        plt.close(fig)
    
    def _save_experiment_figure(self, fig, index):
        viz_folder_path = os.path.join(self.folder_path, "visualizations")
        os.makedirs(viz_folder_path, exist_ok=True)
        viz_path = os.path.join(viz_folder_path, f"{index}.png")
        fig.savefig(viz_path, format='png', bbox_inches="tight")
        plt.close(fig)
