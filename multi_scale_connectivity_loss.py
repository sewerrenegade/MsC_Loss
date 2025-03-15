import numpy as np
import time
from MsC_Loss import multi_scale_connectivity_loss_helper_functions as helper_functions


class LazyModuleMeta(type):
    """Metaclass that ensures torch.nn.Module is only loaded when needed."""
    
    def __new__(cls, name, bases, dct):
        if not any(issubclass(base, object) for base in bases):
            import importlib
            torch_nn = importlib.import_module("torch.nn")
            bases = (torch_nn.Module,)  # Set the correct base class dynamically
        return super().__new__(cls, name, bases, dct)

class LazyTorchModule(metaclass=LazyModuleMeta):
    """Base class that lazily loads torch.nn.Module when instantiated."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class TopologicalZeroOrderLoss(LazyTorchModule):
    """Topological signature."""
    LOSS_ORDERS = [1,2]
    PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS =["match_scale_order","shallow","moor_method","modified_moor_method","deep"]
    SCALE_MATCHING_METHODS = ["order","distribution","similarity_1","similarity_2","similarity_3","similarity_4"]

    def __init__(self,method="deep",p=2,timeout = 5,multithreading = True, scale_matching_method = "order",importance_scale_fraction_taken=1.0,importance_calculation_strat = None,match_scale_in_space =1,augmentation_factor = 0.0):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        self.perform_lazy_imports()
        super().__init__()
        assert p in TopologicalZeroOrderLoss.LOSS_ORDERS
        self.name = method
        self.p = p
        self.signature_calculator = self.ConnectivityEncoderCalculator
        self.loss_fnc = self.get_torch_p_order_function()
        self.topo_feature_loss = self.get_topo_feature_approach(method)
        self.scale_matching_fnc = self.get_scale_matching_fnc(scale_matching_method)
        self.augmentation_factor = augmentation_factor # positive value
        self.importance_scale_fraction_taken = importance_scale_fraction_taken
        self.calculate_all_losses = False
        self.importance_calculation_strat = importance_calculation_strat
        self.timeout = timeout
        self.multithreading= multithreading
        self.match_scale_in_space= match_scale_in_space
        print(f"Matching scale in Space:{self.match_scale_in_space}")
        self.loss_input_1_2 = True
        self.input_1_req_grad = False
        self.input_2_req_grad = False
        self.setup_multithreading()

        
        
    def get_scale_matching_fnc(self,scale_matching_method):
        assert scale_matching_method in TopologicalZeroOrderLoss.SCALE_MATCHING_METHODS
        self.scale_matching_method = scale_matching_method
        if self.scale_matching_method == TopologicalZeroOrderLoss.SCALE_MATCHING_METHODS[0]:
            scale_fn =  helper_functions.match_scale_order
        elif self.scale_matching_method == TopologicalZeroOrderLoss.SCALE_MATCHING_METHODS[1]:
            scale_fn =  helper_functions.match_scale_distirbution
        elif self.scale_matching_method == TopologicalZeroOrderLoss.SCALE_MATCHING_METHODS[2]:
            scale_fn =  helper_functions.match_scale_on_similarity_1
        elif self.scale_matching_method == TopologicalZeroOrderLoss.SCALE_MATCHING_METHODS[3]:
            scale_fn =  helper_functions.match_scale_on_similarity_2
        elif self.scale_matching_method == TopologicalZeroOrderLoss.SCALE_MATCHING_METHODS[4]:
            scale_fn = helper_functions.match_scale_on_similarity_3
        elif self.scale_matching_method == TopologicalZeroOrderLoss.SCALE_MATCHING_METHODS[5]:
            scale_fn = helper_functions.match_scale_on_similarity_4
        print(f"Using {self.scale_matching_method} to calculate scale matching")
        return scale_fn
        
    def perform_lazy_imports(self):
        from .multi_scale_connectivity_encoder import ConnectivityEncoderCalculator
        from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
        from math import floor
        from random import shuffle, sample
        from torch import stack, tensor, Tensor, long, abs

        # Store imports as instance attributes
        self.ConnectivityEncoderCalculator = ConnectivityEncoderCalculator
        self.con_fututre_ProcessPoolExecutor = ProcessPoolExecutor
        self.con_fututre_as_completed = as_completed
        self.con_fututre_TimeoutError = TimeoutError
        self.math_floor = floor
        self.random_shuffle = shuffle
        self.random_sample = sample
        self.torch_stack = stack
        self.torch_tensor = tensor
        self.torch_Tensor = Tensor
        self.torch_long = long
        self.torch_abs = abs
        
    def select_important_scales(self,topo_encoding_space):
        n_top = int(len(topo_encoding_space.component_total_importance_score) * self.importance_scale_fraction_taken)
        values_array = np.array(topo_encoding_space.component_total_importance_score)

        sorted_indices = np.argsort(values_array)[::-1]

        cutoff_value = values_array[sorted_indices[n_top - 1]]
        eligible_indices = [i for i, v in enumerate(topo_encoding_space.component_total_importance_score) if v == cutoff_value]

        good_indices = [i for i, v in enumerate(topo_encoding_space.component_total_importance_score) if v > cutoff_value]
        if len(eligible_indices) > n_top:
            random_subsmaple_of_eligible_indices = self.random_sample(eligible_indices, n_top - len(good_indices)) # type: ignore 
        else:
            random_subsmaple_of_eligible_indices = eligible_indices
        good_indices.extend(random_subsmaple_of_eligible_indices)
        return list(good_indices)

    def should_calculate_loss(self,distances2):
        if distances2.requires_grad:
            if self.loss_input_1_2:
                self.input_2_req_grad = True
            else:
                self.input_1_req_grad = True
        return (self.loss_input_1_2 and self.input_2_req_grad) or (not self.loss_input_1_2 and self.input_1_req_grad) or self.calculate_all_losses
    def deep_connectivity_loss_of_s1_on_s2(self, topo_encoding_space_1, distances1, topo_encoding_space_2, distances2):
        if self.should_calculate_loss(distances2=distances2):            
            self.stop_event.clear()
            stats = {}
            nb_of_persistent_pairs = len(topo_encoding_space_1.persistence_pairs)
            subdivided_list = self.prepare_topo_feature_subdivision(topo_encoding_space_1)
            start_time = time.time()

            if self.multithreading:
                important_edges_for_each_scale, completed,std_of_workload_across_threads = self.calculate_topo_targets_multithread(topo_encoding_space_1,topo_encoding_space_2,subdivided_list)
            else:
                important_edges_for_each_scale = helper_functions.deep_topo_loss_at_scale(topo_encoding_space_1,topo_encoding_space_2,subdivided_list[0],self.stop_event,self.scale_matching_fnc,self.match_scale_in_space)
                completed = len(important_edges_for_each_scale)
                std_of_workload_across_threads = 0.0 
            pull_loss,push_loss,pairwise_distances_influenced,set_of_unique_edges_influenced,nb_pulled_edges,nb_pushed_edges,scale_demographic_infos = self.generate_diff_loss_from_topo_targets(important_edges_for_each_scale,topo_encoding_space_1,topo_encoding_space_2,distances2)
            total_time_section = time.time() - start_time
            return self.aggregate_loss_and_generate_log(pull_loss,push_loss,pairwise_distances_influenced,set_of_unique_edges_influenced,nb_pulled_edges,nb_pushed_edges,scale_demographic_infos,completed,nb_of_persistent_pairs,total_time_section,std_of_workload_across_threads,distances2)
        else:
            return self.torch_tensor(0.0, device=distances2.device,requires_grad=True),{} # type: ignore      
            

    
    def moor_method_calculate_loss_of_s1_on_s2(self,topo_encoding_space_1,distances1,topo_encoding_space_2,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []
        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]]
            scale_of_edge_in_space_2 = distances2[topo_encoding_space_1.persistence_pairs[index][0],topo_encoding_space_1.persistence_pairs[index][1]]
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = self.torch_stack(differentiable_scale_of_equivalent_edges_in_space_1) # type: ignore 
        differentiable_scale_of_equivalent_edges_in_space_2 = self.torch_stack(differentiable_scale_of_equivalent_edges_in_space_2)# type: ignore 
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)
    
    def modified_moor_method_calculate_loss_of_s1_on_s2(self,topo_encoding_space_1,distances1,topo_encoding_space_2,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []

        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]]
            scale_of_edge_in_space_2 = distances2[topo_encoding_space_1.persistence_pairs[index][0],topo_encoding_space_1.persistence_pairs[index][1]]
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = self.torch_stack(differentiable_scale_of_equivalent_edges_in_space_1)/ topo_encoding_space_1.distance_of_persistence_pairs[-1] # type: ignore 
        differentiable_scale_of_equivalent_edges_in_space_2 = self.torch_stack(differentiable_scale_of_equivalent_edges_in_space_2)/ topo_encoding_space_2.distance_of_persistence_pairs[-1] # type: ignore 
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)
    
    def shallow_connecitivty_loss_of_s1_on_s2(self,topo_encoding_space_1,distances1,topo_encoding_space_2,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []
        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            equivalent_feature_in_space_2 = topo_encoding_space_2.what_connected_these_two_points(edge_indices[0], edge_indices[1])
            equivalent_edge_in_space_2 = equivalent_feature_in_space_2["persistence_pair"]
            scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]] / topo_encoding_space_1.distance_of_persistence_pairs[-1]
            scale_of_edge_in_space_2 = distances2[equivalent_edge_in_space_2[0], equivalent_edge_in_space_2[1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = self.torch_stack(differentiable_scale_of_equivalent_edges_in_space_1) # type: ignore 
        differentiable_scale_of_equivalent_edges_in_space_2 = self.torch_stack(differentiable_scale_of_equivalent_edges_in_space_2) # type: ignore 
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)

    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        nondiff_distances1 = self.preprocess_dist_mat(distances1)
        nondiff_distances2 = self.preprocess_dist_mat(distances2)
        topo_encoding_space_1 = self.calulate_space_connectivity_encoding(nondiff_distances1)
        topo_encoding_space_2 = self.calulate_space_connectivity_encoding(nondiff_distances2)
        self.loss_input_1_2 = True
        loss_2_on_1 = self.topo_feature_loss(topo_encoding_space_1=topo_encoding_space_1,
                                                      distances1=distances1,
                                                      topo_encoding_space_2=topo_encoding_space_2,
                                                      distances2=distances2)
        self.loss_input_1_2 = False
        loss_1_on_2 = self.topo_feature_loss(topo_encoding_space_1=topo_encoding_space_2,
                                                      distances1=distances2,
                                                      topo_encoding_space_2=topo_encoding_space_1,
                                                      distances2=distances1)
        loss,log = TopologicalZeroOrderLoss.combine_topo_feature_loss_function_outputs(loss_1_on_2,loss_2_on_1)

        return loss,log

    @staticmethod
    def combine_topo_feature_loss_function_outputs(topo_loss_1_on_2,topo_loss_2_on_1):
        topo_loss_1_on_2 = TopologicalZeroOrderLoss.extract_topo_feature_loss_function_output(topo_loss_1_on_2)
        topo_loss_2_on_1 = TopologicalZeroOrderLoss.extract_topo_feature_loss_function_output(topo_loss_2_on_1)
        log = {f"{key}_1_on_2":value for key,value in topo_loss_1_on_2[1].items()}
        log.update({f"{key}_2_on_1":value for key,value in topo_loss_2_on_1[1].items()})
        log["topo_loss_1_on_2"] =float(topo_loss_1_on_2[0].item())
        log["topo_loss_2_on_1"] =float(topo_loss_2_on_1[0].item())

        combined_loss = topo_loss_1_on_2[0] + topo_loss_2_on_1[0]
        return combined_loss,log
    @staticmethod
    def extract_topo_feature_loss_function_output(topo_output):
        if isinstance(topo_output, tuple):
            log_info = topo_output[1]
            loss = topo_output[0]
        else:
            log_info = {}
            loss = topo_output
        return loss,log_info 

    def get_topo_feature_approach(self,method):
        self.method = self.set_scale_matching_method(method)
        if self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[0]:
            topo_fnc =  None #depricated
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[1]:
            topo_fnc =  self.shallow_connecitivty_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[2]:
            topo_fnc =  self.moor_method_calculate_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[3]:
            topo_fnc =  self.modified_moor_method_calculate_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[4]:
            topo_fnc = self.deep_connectivity_loss_of_s1_on_s2
        print(f"Using {self.method} to calculate per topo feature loss")
        return topo_fnc

    def set_scale_matching_method(self,scale_matching_method):
        if scale_matching_method in TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS:
            return scale_matching_method
        else:
            raise ValueError(f"Scale matching methode {scale_matching_method} does not exist")

    def preprocess_dist_mat(self,D):
        D = self.to_numpy(D)
        D = self.augment_distance_matrix(D)
        D = self.perturb_distance_matrix(D)
        return D
    
    def augment_distance_matrix(self,D):
        if self.augmentation_factor != 0.0:
            high = self.augmentation_factor + 1
            low = 1 - self.augmentation_factor
            upper_triangle = np.random.rand(*D.shape) * (high - low) + low
            symmetric_matrix = np.triu(upper_triangle) + np.triu(upper_triangle, 1).T
            np.fill_diagonal(symmetric_matrix, 1)
            
            return D * symmetric_matrix
        else:
            return D
    
    def calulate_space_connectivity_encoding(self,distance_matrix):
        topo_encoder = self.signature_calculator(distance_matrix,importance_calculation_strat=self.importance_calculation_strat)
        topo_encoder.calculate_connectivity()
        return topo_encoder

    # using numpy to make sure autograd of torch is not disturbed

    def to_numpy(self,obj):
        if isinstance(obj, np.ndarray):
            # If it's already a NumPy array, return as is
            return obj
        elif isinstance(obj, self.torch_Tensor): # type: ignore 
            # Detach the tensor from the computation graph, move to CPU if necessary, and convert to NumPy
            return obj.detach().cpu().numpy()
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor")

    def perturb_distance_matrix(self,D, epsilon=1e-10):
        """
        Ensures that no two entries in the strict upper (or lower) triangular part of the distance matrix are the same.

        Args:
            D (np.ndarray): Symmetric distance matrix of shape (N, N) with zeros on the diagonal.
            epsilon (float): Smallest amount of noise added to break ties.

        Returns:
            np.ndarray: Perturbed distance matrix.
        """
        assert D.shape[0] == D.shape[1], "D must be a square matrix"
        assert np.allclose(D, D.T), "D must be symmetric"
        assert np.all(np.diag(D) == 0), "Diagonal elements must be zero"

        N = D.shape[0]
        upper_tri_indices = np.triu_indices(N, k=1)
        upper_values = D[upper_tri_indices]

        # Find duplicates in upper triangle
        unique, inverse, counts = np.unique(upper_values, return_inverse=True, return_counts=True)
        duplicate_mask = counts[inverse] > 1  # Mask of duplicate values in upper_values

        # Apply perturbation only to duplicates
        perturbation = (np.random.rand(len(upper_values)) - 0.5) * epsilon
        upper_values[duplicate_mask] += perturbation[duplicate_mask]

        # Assign back to the upper triangle and reflect to lower triangle
        D[upper_tri_indices] = upper_values
        D.T[upper_tri_indices] = upper_values  # Reflect for symmetry

        return D
        
    def prepare_topo_feature_subdivision(self, topo_encoding_space_1):
        shuffled_indices_of_topo_features = self.select_important_scales(topo_encoding_space_1)
        self.random_shuffle(shuffled_indices_of_topo_features)  # type: ignore
        k, m = divmod(len(shuffled_indices_of_topo_features), self.available_threads + 1)
        subdivided_list = [shuffled_indices_of_topo_features[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(self.available_threads + 1)]
        return subdivided_list
    def calculate_topo_targets_multithread(self,topo_encoding_space_1,topo_encoding_space_2,subdivided_list):
        important_edges_for_each_scale = []
        args = [(topo_encoding_space_1,topo_encoding_space_2,indices,self.stop_event,self.scale_matching_fnc.__name__,self.match_scale_in_space) for indices in subdivided_list[:-1]]
        futures = {self.executor.submit(helper_functions.deep_topo_loss_at_scale, *arg): i for i, arg in enumerate(args)}
        self.main_thread_event.clear()
        main_thread_execution_result =  helper_functions.deep_topo_loss_at_scale(topo_encoding_space_1,topo_encoding_space_2,subdivided_list[-1],self.main_thread_event,self.scale_matching_fnc,self.match_scale_in_space)
        self.stop_event.set()#this us useful if using threading.event structure
        completed = 0
        threads_that_returned = 0
        nb_of_complete_per_thread = []
        try:
            for future in self.con_fututre_as_completed(futures,timeout=self.timeout): # type: ignore # this extra timeout is the amount of extra time it will w8 for last iteration to finish, otherwise it will dump all the results
                try:
                    result = future.result()
                    important_edges_for_each_scale.extend(result)
                    completed = completed + len(result)
                    nb_of_complete_per_thread.append(len(result))
                    threads_that_returned = threads_that_returned + 1
                except Exception as e:
                    print(f"Error occurred while getting result: {e}")
        except TimeoutError:
            print(f"Called for stopping of threads, however {len(futures) - threads_that_returned} out of {len(futures) + 1} have failed to terminate within the timeout window of {self.timeout}sec. Throwing results of failed threads and continuing.")
        important_edges_for_each_scale.extend(main_thread_execution_result)
        completed = completed + len(main_thread_execution_result)
        nb_of_complete_per_thread.append(len(main_thread_execution_result))
        std_of_workload_across_threads = np.std(nb_of_complete_per_thread,ddof=1)/np.sum(nb_of_complete_per_thread) if len(nb_of_complete_per_thread)>1 else 0.0
        return important_edges_for_each_scale, completed, std_of_workload_across_threads

    def generate_diff_loss_from_topo_targets(self,important_edges_for_each_scale,topo_encoding_space_1,topo_encoding_space_2,distances2):
        pairwise_distances_influenced = 0
        nb_pulled_edges = 0
        nb_pushed_edges = 0
        set_of_unique_edges_influenced = set()
        push_loss = self.torch_tensor(0.0, device=distances2.device)# type: ignore 
        pull_loss = self.torch_tensor(0.0, device=distances2.device)# type: ignore 
        scale_demographic_infos = []
        for scale,pull_edges,push_edges,scale_index in important_edges_for_each_scale:
            all_edges = pull_edges + push_edges
            set_of_unique_edges_influenced.update(set(all_edges))
            if len(all_edges) == 0:
                continue
            push_important_pairs_tensor = self.torch_tensor(np.array(push_edges), dtype=self.torch_long, device=distances2.device)# type: ignore 
            pull_important_pairs_tensor = self.torch_tensor(np.array(pull_edges), dtype=self.torch_long, device=distances2.device)# type: ignore 
            scale_demographic_info = [scale,0.0,0.0] #scale,pull,push
            scale = self.torch_tensor(scale, device=distances2.device)# type: ignore 
            if len(pull_edges) != 0:
                pull_selected_diff_distances = distances2[pull_important_pairs_tensor[:, 0], pull_important_pairs_tensor[:, 1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]
                pull_loss_at_this_scale = abs(pull_selected_diff_distances - scale) ** self.p
                pull_loss_at_this_scale = pull_loss_at_this_scale.sum()
                scale_demographic_info[1] = float(pull_loss_at_this_scale.item())
                pull_loss = pull_loss + pull_loss_at_this_scale*topo_encoding_space_1.component_total_importance_score[scale_index]                     
            if len(push_edges) != 0:
                push_selected_diff_distances = distances2[push_important_pairs_tensor[:, 0], push_important_pairs_tensor[:, 1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]
                push_loss_at_this_scale = abs(push_selected_diff_distances - scale) ** self.p
                push_loss_at_this_scale = push_loss_at_this_scale.sum()
                scale_demographic_info[2] = float(push_loss_at_this_scale.item())
                push_loss = push_loss + push_loss_at_this_scale*topo_encoding_space_1.component_total_importance_score[scale_index]

            scale_demographic_infos.append(scale_demographic_info)
            pairwise_distances_influenced = pairwise_distances_influenced + len(all_edges)
            nb_pulled_edges = nb_pulled_edges + len(pull_edges)
            nb_pushed_edges = nb_pushed_edges + len(push_edges)
        return pull_loss,push_loss,pairwise_distances_influenced,set_of_unique_edges_influenced,nb_pulled_edges,nb_pushed_edges,scale_demographic_infos
    def aggregate_loss_and_generate_log(self,pull_loss,push_loss,pairwise_distances_influenced,set_of_unique_edges_influenced,nb_pulled_edges,nb_pushed_edges,scale_demographic_infos,completed,nb_of_persistent_pairs,total_time_section,std_of_workload_across_threads,distances2):

            #print(f"Total time take for topology calculatation {total_time_section:.4f} seconds, nb of pers_pairs: {nb_of_persistent_pairs} of which {completed} where calculated, with {pairwise_distances_influenced} paris influenced ")
            if pairwise_distances_influenced > 0:
                completed_safe = completed if completed != 0 else 1  # Avoid division by zero
                nb_of_persistent_pairs_safe = nb_of_persistent_pairs if nb_of_persistent_pairs != 0 else 1  
                total_time_section_safe = total_time_section if total_time_section != 0 else 1  
                
                loss = (push_loss + pull_loss) / (completed_safe * nb_of_persistent_pairs_safe) \
                    if completed != 0 else self.torch_tensor(0.0, device=distances2.device, requires_grad=True)  # type: ignore
                
                topo_step_stats = {
                    "topo_time_taken": float(total_time_section),
                    "nb_of_persistent_edges": nb_of_persistent_pairs,
                    "percentage_toporeg_calc": 100 * float(completed_safe / nb_of_persistent_pairs_safe),
                    "pull_push_ratio": float(nb_pulled_edges / (0.01 + nb_pushed_edges)) if nb_pushed_edges != 0 else -1.0,
                    "nb_pairwise_distance_influenced": pairwise_distances_influenced,
                    "nb_unique_pairwise_distance_influenced": len(set_of_unique_edges_influenced),
                    "rate_of_scale_calculation": float(completed_safe) / float(total_time_section_safe),
                    "pull_push_loss_ratio": pull_loss.item() / push_loss.item() if push_loss.item() != 0 else -1.0,
                    "scale_loss_info": scale_demographic_infos,
                    "std_of_workload_across_threads": std_of_workload_across_threads
                }
            else:
                loss = self.torch_tensor(0.0, device=distances2.device,requires_grad=True) # type: ignore 
                topo_step_stats = {"topo_time_taken": float(total_time_section),"nb_of_persistent_edges":nb_of_persistent_pairs,
                                    "percentage_toporeg_calc":100*float(completed/ nb_of_persistent_pairs),
                                    "nb_pairwise_distance_influenced":pairwise_distances_influenced,"nb_unique_pairwise_distance_influenced":len(set_of_unique_edges_influenced)}
                
            return loss ,topo_step_stats
    
    def get_torch_p_order_function(self):
        import torch.nn as nn
        if self.p ==1 :
            return nn.L1Loss()
        elif self.p == 2:
            return nn.MSELoss()
        else:
            raise ValueError(f"This loss {self.p} is not supported")
        
    def setup_multithreading(self):
        if self.multithreading and self.method == "deep":
            self.available_threads = self.math_floor(self.get_thread_count() * 0.5) # type: ignore #take up 50% of available threads excluding the current one or the one this is executing on
            if self.available_threads == 0:
                self.multithreading = False
                self.stop_event = Timer(self.timeout)
                
            else:
                self.executor =  self.con_fututre_ProcessPoolExecutor(max_workers=self.available_threads)# type: ignore 
                self.stop_event = Timer(self.timeout) #threading.Event()
                self.main_thread_event = Timer(self.timeout*0.9)
                try:
                    import wandb
                    wandb.run.summary["threads_used_for_topo_calc"] = self.available_threads
                except Exception as e:
                    print(f"Could not log threads used by deep topology regularization, probably because the loss is being used standalone, available threads: {self.available_threads}; error:{e}")
            print(f"Available threads : {self.available_threads}")
        else:
            self.available_threads = 0
            self.multithreading = False
            self.stop_event = Timer(self.timeout)    
    def get_thread_count(self):
        import os
        import subprocess
        import psutil
        
        def get_allocated_threads_by_slurm():
            try:
                # Get the number of threads per core from lscpu
                lscpu_output = subprocess.check_output("lscpu | awk '/Thread\\(s\\) per core:/ {print $4}'", shell=True)
                threads_per_core = int(lscpu_output.decode().strip())
                
                # Get the Slurm allocated CPUs
                cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", None))  # Default to 1 if not in Slurm
                allocated_threads = threads_per_core * cpus_per_task
                return allocated_threads
            except Exception as e:
                print(f"Could not detect SLURM environment, probably running on personal hardware; Error msg {e}")
                return None

        def get_threads_from_hardware():
                try:
                    # Get total number of logical processors (threads)
                    threads = psutil.cpu_count(logical=True)
                    return threads
                except Exception as e:
                    print(f"Using only one thread to calculate topology, could not detect how many threads available; Error: {e}")
                    return 1 #assume only one thread exists
                
        threads_alloc_by_slurm = get_allocated_threads_by_slurm()
        if threads_alloc_by_slurm is None:
            threads_available_on_hardware = get_threads_from_hardware()
            return threads_available_on_hardware
        return get_allocated_threads_by_slurm()
    


class Timer:
    def __init__(self, timeout=2, start_time = None):
        self.timeout = timeout
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
    def clear(self):
        self.start_time = time.time()
    def set(self):
        pass #this is nothing, just to mkae it compatible with mutithreading approach
    def is_set(self):
        """Check if the timer has exceeded the set timeout."""
        return (time.time() - self.start_time) > self.timeout
        
    def __getstate__(self):
        # Prepare the state dictionary, including start_time and timeout
        return {'timeout': self.timeout, 'start_time': self.start_time}

    def __setstate__(self, state):
        # Restore the object's state exactly as it was
        self.timeout = state['timeout']
        self.start_time = state['start_time']
    