import numpy as np




# from .multi_scale_connectivity_encoder import ConnectivityEncoderCalculator
# from concurrent.futures import ProcessPoolExecutor,as_completed,TimeoutError
# from math import floor
# from random import shuffle,sample
# from torch import stack,tensor,Tensor,long,abs
# import torch.nn as nn

def count_dict_entries(data: dict) -> tuple[int, int]:
    """
    Takes a dictionary where values are lists and returns a tuple:
    - First element: number of key-value pairs.
    - Second element: total count of elements across all lists.

    Args:
        data (dict): A dictionary where values are lists.

    Returns:
        tuple[int, int]: (number of key-value pairs, total elements in all lists)
    """
    num_pairs = len(data)
    total_elements = sum(len(v) for v in data.values() if isinstance(v, list))
    return num_pairs, total_elements
def match_scale_order(index_in_s1,cs1,cs2):
    index_in_s2 = index_in_s1
    return index_in_s2
def match_scale_distirbution(index_in_s1,cs1,cs2):
    scale_in_s1 = cs1.scales[index_in_s1]
    index_in_s2 = cs2.get_index_of_scale_closest_to(scale_in_s1)
    return index_in_s2

def match_scale_on_similarity_1(index_in_s1,cs1,cs2):
    return match_scale_on_similarity(index_in_s1,cs1,cs2,optimization_fnc_1)
def match_scale_on_similarity_2(index_in_s1,cs1,cs2):
    return match_scale_on_similarity(index_in_s1,cs1,cs2,optimization_fnc_2)
def match_scale_on_similarity_3(index_in_s1,cs1,cs2):
    return match_scale_on_similarity(index_in_s1,cs1,cs2,optimization_fnc_3)
def match_scale_on_similarity_4(index_in_s1,cs1,cs2):
    return match_scale_on_similarity(index_in_s1,cs1,cs2,optimization_fnc_4)
    

def match_scale_on_similarity(index_in_s1,cs1,cs2,opt_fnc):
    component_birth_in_s1_due_to_pers_pair = cs1.get_component_birthed_at_index(index_in_s1)
    size_of_birthed_component = len(component_birth_in_s1_due_to_pers_pair)
    potential_components_of_interest = [cs2.get_components_that_contain_these_points_at_this_scale_index(
                    relevant_points=component_birth_in_s1_due_to_pers_pair, index_of_scale=index_in_s2 
                ) for index_in_s2 in range(len(cs2.scales))]
    nb_of_groups_to_nb_of_intruders = [(len(data),sum(v.size for v in data.values() if isinstance(v, np.ndarray))-size_of_birthed_component) for data in potential_components_of_interest]
    opt_values = [opt_fnc(nb_groups,nb_intruders) for nb_groups,nb_intruders in nb_of_groups_to_nb_of_intruders]
    min_value = min(opt_values)
    index_in_s2 = opt_values.index(min_value)
    return index_in_s2
    #print(nb_of_groups_to_nb_of_intruders)
def optimization_fnc_1(nb_groups,nb_of_intruders):
    return nb_groups+nb_of_intruders
def optimization_fnc_2(nb_groups,nb_of_intruders):
    return (nb_groups-1)**2+nb_of_intruders**2
def optimization_fnc_3(nb_groups,nb_of_intruders):
    return max(nb_groups-1,nb_of_intruders)
def optimization_fnc_4(nb_groups,nb_of_intruders):
    return nb_groups*(nb_of_intruders+1)    
    
def deep_topo_loss_at_scale(topo_encoding_space_1,topo_encoding_space_2,s1_scale_indices,stop_event,scale_matching_strat_fn,match_scale_in_space = 1):
        scale_edges_pairings = []
        for s1_scale_index in s1_scale_indices:
            if not stop_event.is_set():
                if isinstance(scale_matching_strat_fn,str):
                    scale_matching_strat_fn = globals()[scale_matching_strat_fn]
                assert isinstance(s1_scale_index,int)
                assert 0<= s1_scale_index< len(topo_encoding_space_1.scales)
                component_birth_in_s1_due_to_pers_pair = topo_encoding_space_1.get_component_birthed_at_index(s1_scale_index)
                scale_in_s1 = topo_encoding_space_1.scales[s1_scale_index]
                index_of_scale_in_s2 = scale_matching_strat_fn(index_in_s1 = s1_scale_index,cs1 = topo_encoding_space_1,cs2 = topo_encoding_space_2) # topo_encoding_space_2.get_index_of_scale_closest_to(scale_in_s1) # #  
                scale_in_s2 = topo_encoding_space_2.scales[index_of_scale_in_s2]
                relevant_sets_in_s2 = topo_encoding_space_2.get_components_that_contain_these_points_at_this_scale_index(
                    relevant_points=component_birth_in_s1_due_to_pers_pair, index_of_scale=index_of_scale_in_s2 
                )
                to_push_out_at_this_scale = []
                healthy_subsets = []

                for component_in_s2_name, member_vertices in relevant_sets_in_s2.items():
                    good_vertices = np.intersect1d(member_vertices, component_birth_in_s1_due_to_pers_pair)
                    for vertex in member_vertices:
                        if vertex not in component_birth_in_s1_due_to_pers_pair:
                            pair_info = topo_encoding_space_2.what_connected_this_point_to_this_set(
                                point=vertex, vertex_set=good_vertices
                            )["persistence_pair"]
                            to_push_out_at_this_scale.append(pair_info)
                    healthy_subsets.append(good_vertices)#tensor(good_vertices, dtype=long, device=distances2.device)
                if len(healthy_subsets) > 1:
                    pairs_to_pull = topo_encoding_space_2.what_edges_needed_to_connect_these_components(healthy_subsets)
                else:
                    pairs_to_pull = []
                unique_to_push_out_at_this_scale = list(set(to_push_out_at_this_scale))
                scale_edges_pairings.append((scale_in_s1 if match_scale_in_space==1 else scale_in_s2, pairs_to_pull,unique_to_push_out_at_this_scale,s1_scale_index))
            else:
                return scale_edges_pairings
        return scale_edges_pairings
    
    
    # def deep_connectivity_loss_of_s1_on_s2(self, topo_encoding_space_1, distances1, topo_encoding_space_2, distances2,scale_matching_strat = helper_functions.match_scale_order):
    #     if distances2.requires_grad or self.calculate_all_losses:            
    #         self.stop_event.clear()
    #         nb_of_persistent_pairs = len(topo_encoding_space_1.persistence_pairs)
    #         shuffled_indices_of_topo_features = self.select_important_scales(topo_encoding_space_1)
    #         self.random_shuffle(shuffled_indices_of_topo_features) # type: ignore 
    #         k, m = divmod(len(shuffled_indices_of_topo_features), self.available_threads + 1)
    #         subdivided_list = [shuffled_indices_of_topo_features[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(self.available_threads + 1)]
    #         important_edges_for_each_scale = []
    #         start_time = time.time()

    #         if self.multithreading:
    #             args = [(topo_encoding_space_1,topo_encoding_space_2,indices,self.stop_event) for indices in subdivided_list[:-1]]
    #             futures = {self.executor.submit(helper_functions.deep_topo_loss_at_scale, *arg): i for i, arg in enumerate(args)}
    #             self.main_thread_event.clear()
    #             main_thread_execution_result =  helper_functions.deep_topo_loss_at_scale(topo_encoding_space_1,topo_encoding_space_2,subdivided_list[-1],self.main_thread_event)
    #             self.stop_event.set()#this us useful if using threading.event structure
    #             completed = 0
    #             threads_that_returned = 0
    #             nb_of_complete_per_thread = []
    #             try:
    #                 for future in self.con_fututre_as_completed(futures,timeout=self.timeout): # type: ignore # this extra timeout is the amount of extra time it will w8 for last iteration to finish, otherwise it will dump all the results
    #                     try:
    #                         result = future.result()
    #                         important_edges_for_each_scale.extend(result)
    #                         completed = completed + len(result)
    #                         nb_of_complete_per_thread.append(len(result))
    #                         threads_that_returned = threads_that_returned + 1
    #                     except Exception as e:
    #                         print(f"Error occurred while getting result: {e}")
    #             except TimeoutError:
    #                 print(f"Called for stopping of threads, however {len(futures) - threads_that_returned} out of {len(futures) + 1} have failed to terminate within the timeout window of {self.timeout}sec. Throwing results of failed threads and continuing.")
    #             important_edges_for_each_scale.extend(main_thread_execution_result)
    #             completed = completed + len(main_thread_execution_result)
    #             nb_of_complete_per_thread.append(len(main_thread_execution_result))
    #             std_of_workload_across_threads = np.std(nb_of_complete_per_thread,ddof=1)/np.sum(nb_of_complete_per_thread) if len(nb_of_complete_per_thread)>1 else 0.0
    #         else:
    #             important_edges_for_each_scale = helper_functions.deep_topo_loss_at_scale(topo_encoding_space_1,topo_encoding_space_2,subdivided_list[0],self.stop_event)
    #             completed = len(important_edges_for_each_scale)
    #             std_of_workload_across_threads = 0.0 


    #         pairwise_distances_influenced = 0
    #         nb_pulled_edges = 0
    #         nb_pushed_edges = 0
    #         set_of_unique_edges_influenced = set()
    #         push_loss = self.torch_tensor(0.0, device=distances2.device)# type: ignore 
    #         pull_loss = self.torch_tensor(0.0, device=distances2.device)# type: ignore 
    #         scale_demographic_infos = []
    #         for scale,pull_edges,push_edges,scale_index in important_edges_for_each_scale:
    #             all_edges = pull_edges + push_edges
    #             set_of_unique_edges_influenced.update(set(all_edges))
    #             if len(all_edges) == 0:
    #                 continue
    #             push_important_pairs_tensor = self.torch_tensor(np.array(push_edges), dtype=self.torch_long, device=distances2.device)# type: ignore 
    #             pull_important_pairs_tensor = self.torch_tensor(np.array(pull_edges), dtype=self.torch_long, device=distances2.device)# type: ignore 
    #             scale_demographic_info = [scale,0.0,0.0] #scale,pull,push
    #             scale = self.torch_tensor(scale, device=distances2.device)# type: ignore 
    #             if len(pull_edges) != 0:
    #                 pull_selected_diff_distances = distances2[pull_important_pairs_tensor[:, 0], pull_important_pairs_tensor[:, 1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]
    #                 pull_loss_at_this_scale = abs(pull_selected_diff_distances - scale) ** self.p
    #                 pull_loss_at_this_scale = pull_loss_at_this_scale.sum()
    #                 scale_demographic_info[1] = float(pull_loss_at_this_scale.item())
    #                 pull_loss = pull_loss + pull_loss_at_this_scale*topo_encoding_space_1.component_total_importance_score[scale_index]                     
    #             if len(push_edges) != 0:
    #                 push_selected_diff_distances = distances2[push_important_pairs_tensor[:, 0], push_important_pairs_tensor[:, 1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]
    #                 push_loss_at_this_scale = abs(push_selected_diff_distances - scale) ** self.p
    #                 push_loss_at_this_scale = push_loss_at_this_scale.sum()
    #                 scale_demographic_info[2] = float(push_loss_at_this_scale.item())
    #                 push_loss = push_loss + push_loss_at_this_scale*topo_encoding_space_1.component_total_importance_score[scale_index]

    #             scale_demographic_infos.append(scale_demographic_info)
    #             pairwise_distances_influenced = pairwise_distances_influenced + len(all_edges)
    #             nb_pulled_edges = nb_pulled_edges + len(pull_edges)
    #             nb_pushed_edges = nb_pushed_edges + len(push_edges)
    #         total_time_section = time.time() - start_time
    #         #print(f"Total time take for topology calculatation {total_time_section:.4f} seconds, nb of pers_pairs: {nb_of_persistent_pairs} of which {completed} where calculated, with {pairwise_distances_influenced} paris influenced ")
    #         if pairwise_distances_influenced > 0:
    #             completed_safe = completed if completed != 0 else 1  # Avoid division by zero
    #             nb_of_persistent_pairs_safe = nb_of_persistent_pairs if nb_of_persistent_pairs != 0 else 1  
    #             total_time_section_safe = total_time_section if total_time_section != 0 else 1  
                
    #             loss = (push_loss + pull_loss) / (completed_safe * nb_of_persistent_pairs_safe) \
    #                 if completed != 0 else self.torch_tensor(0.0, device=distances2.device, requires_grad=True)  # type: ignore
                
    #             topo_step_stats = {
    #                 "topo_time_taken": float(total_time_section),
    #                 "nb_of_persistent_edges": nb_of_persistent_pairs,
    #                 "percentage_toporeg_calc": 100 * float(completed_safe / nb_of_persistent_pairs_safe),
    #                 "pull_push_ratio": float(nb_pulled_edges / (0.01 + nb_pushed_edges)) if nb_pushed_edges != 0 else -1.0,
    #                 "nb_pairwise_distance_influenced": pairwise_distances_influenced,
    #                 "nb_unique_pairwise_distance_influenced": len(set_of_unique_edges_influenced),
    #                 "rate_of_scale_calculation": float(completed_safe) / float(total_time_section_safe),
    #                 "pull_push_loss_ratio": pull_loss.item() / push_loss.item() if push_loss.item() != 0 else -1.0,
    #                 "scale_loss_info": scale_demographic_infos,
    #                 "std_of_workload_across_threads": std_of_workload_across_threads
    #             }
    #         else:
    #             loss = self.torch_tensor(0.0, device=distances2.device,requires_grad=True) # type: ignore 
    #             topo_step_stats = {"topo_time_taken": float(total_time_section),"nb_of_persistent_edges":nb_of_persistent_pairs,
    #                                "percentage_toporeg_calc":100*float(completed/ nb_of_persistent_pairs),
    #                                "nb_pairwise_distance_influenced":pairwise_distances_influenced,"nb_unique_pairwise_distance_influenced":len(set_of_unique_edges_influenced)}
                
    #         return loss ,topo_step_stats
    #     else:
    #         return self.torch_tensor(0.0, device=distances2.device,requires_grad=True),{} # type: ignore 