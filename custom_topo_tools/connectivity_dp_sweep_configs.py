    
from models.topology_models.custom_topo_tools.connectivity_dp_experiment import DATASETS


DATASETS
ALL_PERUMTAIONS = {
    "dataset_name": [DATASETS[0],DATASETS[1],DATASETS[2],DATASETS[3]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["sgd", "adam","adamw","adan"],
    "LR": [1.0],
    "normalize_input": [True, False],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size','multiplication'],
    "augmentation_strength": [0.01, 0.1],
    "size_of_data": [100, 200],
    "weight_decay": [0.0, 0.01],
}

LR_OPTIMIZER_SWEEP =  {
    "dataset_name": [DATASETS[2]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["sgd", "adam","adamw","adan"],
    "LR": [0.0001,0.001, 0.01, 0.1, 1, 10],
    "normalize_input": [True],
    "importance_weighting_strat": ["none"],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.0],
}

IMPORTANCE_STRAT_SWEEP =  {
    "dataset_name": [DATASETS[3]], #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["sgd"],
    "LR": [0.1],
    "normalize_input": [True],
    "importance_weighting_strat": ["none", "min",'component_persistence','component_size','multiplication'],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.01],
}

NORMALIZE_INPUT_SWEEP =  {
    "dataset_name": DATASETS, #["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]
    "optimizer_name": ["sgd"],
    "LR": [0.1],
    "normalize_input": [True, False],
    "importance_weighting_strat": ["none"],
    "augmentation_strength": [0.0],
    "size_of_data": [200],
    "weight_decay": [0.01],
}