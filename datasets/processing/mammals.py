import pickle
import numpy as np
import shutil 
from pathlib import Path 

DATASETS = Path("/home/michaeldodds/Projects/manchester/computer_vision/datasets")
RAW_DATASETS = DATASETS / 'raw'
M_RAW_DATASETS = RAW_DATASETS / 'mammals-image-classification-dataset-45-animals/versions/1/mammals'
P_DATASETS = DATASETS / 'processed'
M_P_DATASETS = P_DATASETS / 'mammals'

def create_id_dictionary(directory: Path): 
    files = list(directory.iterdir())
    abs_id = {id + 1: name.name for id, name in enumerate(files)}
    return abs_id, invert_id(abs_id)

def invert_id(dictionary): 
    return {i: j for j, i in dictionary.items()}

def move_files(src, dest: Path, abs_id):
    
    if not dest.exists():
        dest.mkdir(parents=True) 
    
    for key, value in abs_id.items():
        file_src = src / str(value)
        file_dest = dest / str(key)
        if file_dest.exists():
            continue
        shutil.copytree(file_src, file_dest) 


if __name__ == '__main__': 
    abs_id, name_to_id = create_id_dictionary(M_RAW_DATASETS)
    move_files(M_RAW_DATASETS, M_P_DATASETS / 'train', abs_id)
    with open(M_P_DATASETS / 'class_index', 'wb') as fo: 
        pickle.dump(abs_id, fo)