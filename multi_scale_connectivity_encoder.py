import numpy as np

IMPORTANCE_CALCULATION_STRATS = ['component_persistence','component_size','min','multiplication','none','all_dev']
DEFAULT_IMPORTANCE_CALCULATION_STRAT = IMPORTANCE_CALCULATION_STRATS[-2]
#WARNING: none of this is differentiable, torch can not track gradients through this... use this only to calculate and match topological features
class ConnectivityEncoderCalculator:
    def __init__(self, distance_mat,importance_calculation_strat = None):
        assert isinstance(distance_mat, np.ndarray)
        assert distance_mat.shape[0] == distance_mat.shape[1]
        assert len(distance_mat.shape) == 2
        self.distance_matrix = distance_mat
        if importance_calculation_strat is None:
            self.importance_calculation_strat = DEFAULT_IMPORTANCE_CALCULATION_STRAT
        else:
            self.importance_calculation_strat = importance_calculation_strat
            assert self.importance_calculation_strat in IMPORTANCE_CALCULATION_STRATS
        self.n_vertices = distance_mat.shape[0]
        self.topo_scale_evolution =[]
        self._current_topology = np.arange(self.n_vertices, dtype=int)
        self.persistence_pairs = None
        self.scales = None
        self.distance_of_persistence_pairs = None
        self.component_size_importance_score = None
        self.component_persistence_importance_score = None
        self.component_total_importance_score = np.ones(self.n_vertices-1)
        self.persistence_of_components = None

    def get_component_birthed_at_index(self,index):
        state = self.topo_scale_evolution[index]
        pers_pair = self.persistence_pairs[index]
        assert state[pers_pair[0]] == state[pers_pair[1]]
        return np.where(state == state[pers_pair[0]])[0] #assuming 1d array
    
    def get_index_of_scale_closest_to(self, target):
        idx = np.searchsorted(self.scales, target) - 1
        return idx if idx >= 0 else -1

    def what_edges_needed_to_connect_these_components(self,list_of_components:list):
        nb_of_sets = len(list_of_components)
        set_to_edge_mapping = {}
        if nb_of_sets == 1:
            pass
        set_distance_matrix = np.zeros((nb_of_sets, nb_of_sets))
        for i in range(nb_of_sets):
            for j in range(i+1,nb_of_sets):
                edge,distance = self.get_shortest_distance_between_2_sets_ignoring_all_other_points(list_of_components[i],list_of_components[j])
                set_distance_matrix[i,j] = distance
                set_distance_matrix[j,i] = distance
                set_to_edge_mapping[(i,j)] = edge 
        smaller_homology_problem = ConnectivityEncoderCalculator(distance_mat=set_distance_matrix,importance_calculation_strat="none")
        smaller_homology_problem.calculate_connectivity(calculate_importance=False)
        pers_pairs = smaller_homology_problem.persistence_pairs
        real_pers_pairs = [set_to_edge_mapping[pers_pair] for pers_pair in pers_pairs]
        return real_pers_pairs

    def get_shortest_distance_between_2_sets_ignoring_all_other_points(self,set1,set2):
        A, B = np.meshgrid(set1, set2, indexing='ij')
        combinations = np.vstack([A.ravel(), B.ravel()]).T
        distances = self.distance_matrix[combinations[:,0],combinations[:,1]]
        shortest_index = np.argmin(distances)
        shortest_edge = combinations[shortest_index]
        return tuple(np.sort(shortest_edge)),distances[shortest_index]

    def normalize_score(self,score_list):
        if len(score_list) == 0:
            return score_list
        avg_importance = sum(score_list)/len(score_list)
        if avg_importance == 0:
            return score_list
        return [score/avg_importance for score in score_list]
    
    def get_components_that_contain_these_points_at_this_scale_index(self,relevant_points,index_of_scale):
        state_at_scale_index = self.topo_scale_evolution[index_of_scale,:]
        groups_included = {} #the key is the group name (ie point with largest index in comp) and value is a list of included points in component
        for point in relevant_points:
            grp_name = state_at_scale_index[point]
            if not grp_name in groups_included:
                groups_included[grp_name] = np.where(state_at_scale_index == grp_name)[0]
        return groups_included
    
    def encode_connectivity(self):
        """
        Encode the connectivity of a space at each scale as strings, excluding single-point components.

        Args:
            connectivity_list (list of np.ndarray): List of connectivity arrays where each array is of length `n`.

        Returns:
            list of str: List of connectivity representations for each scale, excluding single-point components.
        """
        def array_to_connectivity_string(array):
            """
            Convert a connectivity array to a string representation like {2,3,4},{1},{0,5}, excluding single-point components.

            Args:
                array (np.ndarray): Array encoding connectivity.

            Returns:
                str: Connectivity string.
            """
            components = {}
            for idx, value in enumerate(array):
                leader = int(value)
                if leader not in components:
                    components[leader] = []
                components[leader].append(idx)
            
            # Sort and format the components, excluding single-point ones
            connectivity_parts = []
            for component in components.values():
                if len(component) > 1:  # Exclude single-point components
                    connectivity_parts.append("{" + ",".join(map(str, sorted(component))) + "}")
            
            return ",".join(connectivity_parts)

        return [array_to_connectivity_string(np.array(conn)) for conn in self.topo_scale_evolution]
    
    def calculate_connectivity(self,calculate_importance = True):
        tri_strict_upper_indices = np.triu_indices_from(self.distance_matrix,k=1)
        edge_weights = self.distance_matrix[tri_strict_upper_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')
        persistence_pairs = []
        edge_distances = []
        component_size_importance_scores = []
        lifetime_of_components = []
        group_names = []

        for edge_index, edge_weight in zip(edge_indices, edge_weights[edge_indices]):

            u = tri_strict_upper_indices[0][edge_index]
            v = tri_strict_upper_indices[1][edge_index]

            u_group = self.get_point_current_group(u)
            v_group = self.get_point_current_group(v)
            
            if u_group == v_group:
                continue # no 0 order topological feature created since connectivity remains unchanged
            
            members_of_u_group ,  members_of_v_group = self.merge_groups(u_group, v_group)
            size_importance = min(len(members_of_u_group),len(members_of_v_group)) #(len(members_of_u_group)*len(members_of_v_group))**0.5
            persistence_pairs.append((min(u, v), max(u, v)))
            edge_distances.append(edge_weight)
            component_size_importance_scores.append(size_importance)
            
            lifetime_of_components.append(None)
            birth_index_u,birth_index_v = self.find_last_two_indices(group_names,u_group,v_group)
            if birth_index_u != -1:
                assert lifetime_of_components[birth_index_u] == None
                lifetime_of_components[birth_index_u] = edge_weight - edge_distances[birth_index_u]
            if birth_index_v != -1:
                assert lifetime_of_components[birth_index_v] == None
                lifetime_of_components[birth_index_v] = edge_weight - edge_distances[birth_index_v]
            group_names.append(max(u_group,v_group))
            self.save_current_topo_enc()
            if len(persistence_pairs) == self.n_vertices -1:
                assert lifetime_of_components[-1] == None
                lifetime_of_components[-1] = 0.0
                lifetime_of_components[-1] = max(lifetime_of_components)
                break
        self.scales = [edge_distance/edge_distances[-1] for edge_distance in edge_distances] if edge_distances[-1] != 0 else edge_distances
        self.distance_of_persistence_pairs = edge_distances
        self.persistence_pairs = persistence_pairs
        self.topo_scale_evolution = np.vstack(self.topo_scale_evolution)
        if calculate_importance:
            self.persistence_of_components = lifetime_of_components
            self.component_persistence_importance_score = self.normalize_score(lifetime_of_components)#self.normalize_score([lifetime_of_component/(self.scales[index]+self.scales[int(len(self.scales)*0.3)]) for index,lifetime_of_component in enumerate(self.persistence_of_components)])
            self.component_size_importance_score = self.normalize_score(component_size_importance_scores)
            # importance_strats = self.calculate_importance_score("all_dev")
            # import pandas as pd
            # df = pd.DataFrame(importance_strats)
            # df = df.round(4) 
            # df.to_csv("importance_scores.csv", index=True)
            self.component_total_importance_score = self.calculate_importance_score()
            #self.component_total_importance_score = self.component_size_importance_score
        return self
    
    def calculate_importance_score(self,importance_calculation_strat = None):
        assert not self.component_size_importance_score is None
        assert not self.component_persistence_importance_score is None
        if importance_calculation_strat is None:
            importance_calculation_strat = self.importance_calculation_strat
            
        if importance_calculation_strat == 'min':
            score = [min(self.component_size_importance_score[idx],self.component_persistence_importance_score[idx]) for idx in range(self.n_vertices-1)]
        elif importance_calculation_strat == 'component_persistence':
            score = self.component_persistence_importance_score
        elif importance_calculation_strat == 'component_size':
            score = self.component_size_importance_score
        elif importance_calculation_strat == 'multiplication':
            score = [self.component_size_importance_score[idx]*self.component_persistence_importance_score[idx] for idx in range(self.n_vertices-1)]
        elif importance_calculation_strat == 'none':
            score = [1.0]*(self.n_vertices-1) 
        elif importance_calculation_strat == 'all_dev':
            return  {strat_name: self.normalize_score(self.calculate_importance_score(strat_name)) for strat_name in IMPORTANCE_CALCULATION_STRATS[:-2]}
        return self.normalize_score(score)
    
    def find_last_two_indices(self,lst, int1, int2):
        """
        Finds the indices of the last occurrences of two integers in a list.

        Parameters:
            lst (list): The input list.
            int1 (int): The first integer.
            int2 (int): The second integer.

        Returns:
            tuple: A tuple of indices (index1, index2). If an integer is not found, its index is -1.
        """
        index1 = -1
        index2 = -1

        for i in reversed(range(len(lst))):
            if lst[i] == int1 and index1 == -1:
                index1 = i
            if lst[i] == int2 and index2 == -1:
                index2 = i
            # Stop early if both indices are found
            if index1 != -1 and index2 != -1:
                break

        return index1, index2
    def save_current_topo_enc(self):
        self.topo_scale_evolution.append(np.copy(self._current_topology))
        # x= np.unique(self._current_topology, return_counts=True)
        # stats = {}
        # stats["cluster_ids"] = x[0]
        # stats["cluster_nb_of_memebers"] = x[1]
        # stats["number_of_clusters"] = len(x[0])
        # self.topo_progression_stats.append(stats)
        # self.sanity_checker.append(len(x[0]))


    def get_point_current_group(self, u):
        return self._current_topology[u]

    def merge_groups(self, u_group, v_group):
        members_of_v_group , members_of_u_group = np.where(self._current_topology == v_group),np.where(self._current_topology == u_group)
        if u_group > v_group:
            self._current_topology[members_of_v_group] = u_group
        elif u_group < v_group:
            self._current_topology[members_of_u_group] = v_group
        else:
            print("WTF u doing idiot")
            assert u_group != v_group
        return members_of_u_group[0] ,  members_of_v_group[0]
        
    def what_connected_these_two_points(self,u,v):
        for index,connectivity in enumerate(self.topo_scale_evolution):
            if connectivity[u] == connectivity[v]:
                connecting_index = index
                break
        connecting_info = {"index": connecting_index,"persistence_pair": self.persistence_pairs[connecting_index],"scale": self.scales[connecting_index],"median_order":(connecting_index+1)/len(self.persistence_pairs)}
        return connecting_info
    
    def what_connected_this_point_to_this_set(self,point,vertex_set):
        x = self.topo_scale_evolution[:, point]  # Shape (m,)
        y = self.topo_scale_evolution[:, vertex_set]  # Shape (m, n)

        # Check if any element in y[i] matches x[i] for each row
        matches = (y == x[:, None])  # Broadcasting: matches[i, j] is True if x[i] == y[i, j]

        # Find the first row with any match (along axis 1)
        row_has_match = np.any(matches, axis=1)
        
        # Get the smallest index where a match exists
        try:
            connecting_index = np.argmax(row_has_match) if np.any(row_has_match) else None
        except IndexError:
            connecting_index = None
            
        if connecting_index != -1:
            # Populate connection information only if a connection is found
            connecting_info = {
                "index": connecting_index,
                "persistence_pair": self.persistence_pairs[connecting_index],
                "scale": self.scales[connecting_index],
                "median_order": (connecting_index+1) / len(self.persistence_pairs),
            }
            return connecting_info
        else:
            raise RuntimeError(f"Could not find a scale at which {point} and {vertex_set} connect, which should be impossible")
    
    def what_connected_this_point_to_this_set_old(self,point,set):
        for index,connectivity in enumerate(self.topo_scale_evolution):
            if connectivity[point] in connectivity[set]:
                connecting_index = index - 1 # this is because the s=0 topology encoding is inserted automatically
                break
        connecting_info = {"index": connecting_index,"persistence_pair": self.persistence_pairs[connecting_index],"scale": self.scales[connecting_index],"median_order":(connecting_index+1)/len(self.persistence_pairs)}
        return connecting_info