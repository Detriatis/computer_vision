import pickle
import numpy as np 
from pathlib import Path
import shutil


def load_class_mappings(filepath):
    with open(filepath, 'r') as fp: 
        lines = fp.readlines()
    lines = [line.strip('\n').split('\t') for line in lines]

    class_mappings = {} 
    for ele_id, abs_id in lines: 
        class_mappings[int(abs_id)] = int(ele_id)

    return class_mappings, invert_id(class_mappings)         

def invert_id(dictionary): 
    return {i: j for j, i in dictionary.items()}

def load_image_files(directorypath: Path):
    files = list(directorypath.iterdir())
    identifiers = [file.name.strip("'").split('_') for file in files]
    return identifiers, files

def split(filepath, files):
    
    with open(filepath, 'r') as fp:
        lines = fp.readlines()
    
    file_names = [line.strip('\n').split('\t')[1] for line in lines]
    ids = [line.strip('\n').split('\t')[0] for line in lines]
    
    return ids, file_names 




def save_image_files(srcpath, directorypath, ids, files, ele_mappings): 
    
    for id, file in zip(ids, files): 
        
        id_dir = directorypath / str(ele_mappings[int(id)])
        img_path = id_dir / file

        if not id_dir.exists():
            id_dir.mkdir(parents=True)

        shutil.copy2(srcpath / file, img_path) 


# Paths to directories 
DATASETS = Path("/home/michaeldodds/Projects/manchester/computer_vision/datasets/")
ELPEPHANT_DIR = DATASETS / 'raw/ELPephant'
IMAGES = ELPEPHANT_DIR / 'images'
PROCESSED_ELPEPHANT = DATASETS /'processed/ELPephant'

if not PROCESSED_ELPEPHANT.exists():
    PROCESSED_ELPEPHANT.mkdir()

if __name__ == '__main__':
    abs_mappings, ele_mappings = load_class_mappings(ELPEPHANT_DIR / 'class_mapping.txt')
    ids, files = load_image_files(IMAGES)
    train_ids, train_files = split(ELPEPHANT_DIR / 'train.txt', files)
    test_ids, test_files = split(ELPEPHANT_DIR / 'val.txt', files)
    save_image_files(IMAGES, PROCESSED_ELPEPHANT / 'train', train_ids, train_files, ele_mappings)
    save_image_files(IMAGES, PROCESSED_ELPEPHANT / 'test', test_ids, test_files, ele_mappings)
    
    with open(PROCESSED_ELPEPHANT / 'class_index', 'wb') as fo: 
        pickle.dump(abs_mappings, fo) 