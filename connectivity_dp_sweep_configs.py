    
from .connectivity_dp_experiment import DATASETS


DATASETS
ALL_PERUMTAIONS = {
    "dataset_name": [DATASETS[0],DATASETS[1],DATASETS[2],DATASETS[3]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["sgd", "adam","adamw","adan"],
    "LR": [1.0],
    "normalize_input": [False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size','multiplication'],
    "augmentation_strength": [0.001],
    "size_of_data": [200],
    "weight_decay": [0.0],
}
OPT_PERMUTATIONS_MNSIT = {
    "dataset_name": [DATASETS[0]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["sgd", "adam","adamw"],
    "LR": [1.0,0.5,0.1],
    "normalize_input": [False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size','multiplication'],
    "augmentation_strength": [0.001],
    "size_of_data": [200],
    "weight_decay": [0.0],
}
OPT_PERMUTATIONS_SWISS = {
    "dataset_name": [DATASETS[1]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["sgd", "adam","adamw"],
    "LR": [1.0,0.5,0.1],
    "normalize_input": [False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size','multiplication'],
    "augmentation_strength": [0.001],
    "size_of_data": [200],
    "weight_decay": [0.0],
}
OPT_PERMUTATIONS_DINO = {
    "dataset_name": [DATASETS[2]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["sgd", "adam","adamw"],
    "LR": [1.0,0.5,0.1],
    "normalize_input": [False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size','multiplication'],
    "augmentation_strength": [0.001],
    "size_of_data": [200],
    "weight_decay": [0.0],
}
OPT_PERMUTATIONS_CLUSTERS = {
    "dataset_name": [DATASETS[3]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["sgd", "adam","adamw"],
    "LR": [1.0,0.5,0.1],
    "normalize_input": [False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size','multiplication'],
    "augmentation_strength": [0.001],
    "size_of_data": [200],
    "weight_decay": [0.0],
}
LR_OPTIMIZER_SWEEP =  {
    "dataset_name": DATASETS, #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["sgd", "adam","adamw","adan"],
    "LR": [0.1,0.5, 1,5, 10],
    "normalize_input": [False],
    "importance_weighting_strat": ["none"],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.0],
}
LR_OPTIMIZER_SWEEP_2 =  {
    "dataset_name": DATASETS, #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["adam"],
    "LR": [10, 1, 0.1, 0.01,0.001, 0.0001],
    "normalize_input": [False],
    "importance_weighting_strat": ["min"],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.0],
}

IMPORTANCE_STRAT_SWEEP =  {
    "dataset_name": DATASETS, #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["adam"],
    "LR": [1],
    "normalize_input": [False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size','multiplication'],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.0],
    "p_importance_filter":[1.0,0.75,0.5,0.25]
}

NORMALIZE_INPUT_SWEEP =  {
    "dataset_name": DATASETS, #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["adam"],
    "LR": [1],
    "normalize_input": [True, False],
    "importance_weighting_strat": ["none"],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.0],
}

WEIGHT_DECAY_SWEEP =  {
    "dataset_name": DATASETS, #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["adam"],
    "LR": [1],
    "normalize_input": [False],
    "importance_weighting_strat": ["none"],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.0,0.00001,0.0001,0.001],
}

AUGMENTATION_STRENGTH_SWEEP =  {
    "dataset_name": DATASETS, #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["adam"],
    "LR": [1],
    "normalize_input": [False],
    "importance_weighting_strat": ["none"],
    "augmentation_strength": [0.0,0.001,0.01],
    "size_of_data": [200],
    "weight_decay": [0.0],
}

DATA_SIZE_SWEEP =  {
    "dataset_name": DATASETS, #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["adam"],
    "LR": [1],
    "normalize_input": [False],
    "importance_weighting_strat": ["none"],
    "augmentation_strength": [0.001],
    "size_of_data": [50,100,200,400],
    "weight_decay": [0.0],
}

NEW_PERUMTAIONS_MNIST = {
    "dataset_name": [DATASETS[0]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["adam"],
    "LR": [1.0],
    "normalize_input": [False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size'],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.0],
    "scale_matching_strat":  ["order","distribution","similarity_1","similarity_2","similarity_3","similarity_4"],
    "match_scale_in_space" : [1,2]
}
NEW_PERUMTAIONS_SWISS_ROLL = {
    "dataset_name": [DATASETS[1]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["adam"],
    "LR": [1.0],
    "normalize_input": [False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size'],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.0],
    "scale_matching_strat":  ["order","distribution","similarity_1","similarity_2","similarity_3","similarity_4"],
    "match_scale_in_space" : [1,2]
}
NEW_PERUMTAIONS_DINOBLOOM = {
    "dataset_name": [DATASETS[2]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["adam"],
    "LR": [1.0],
    "normalize_input": [False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size'],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.0],
    "scale_matching_strat":  ["order","distribution","similarity_1","similarity_2","similarity_3","similarity_4"],
    "match_scale_in_space" : [1,2]
}
NEW_PERUMTAIONS_CLUSTERS = {
    "dataset_name": [DATASETS[3]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["adam"],
    "LR": [1.0],
    "normalize_input": [False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size'],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.0],
    "scale_matching_strat":  ["order","distribution","similarity_1","similarity_2","similarity_3","similarity_4"],
    "match_scale_in_space" : [1,2]
}

