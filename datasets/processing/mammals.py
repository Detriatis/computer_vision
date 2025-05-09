import shutil 
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle

DATASETS = Path("/home/REDACTED/Projects/manchester/computer_vision/datasets")
RAW_DATASETS = DATASETS / 'raw'
M_RAW_DATASETS = RAW_DATASETS / 'mammals-image-classification-dataset-45-animals/versions/1/mammals'
# Paths
BASE = Path("/home/REDACTED/Projects/manchester/computer_vision/datasets/processed/mammals")
SRC  =  M_RAW_DATASETS 
DST_TRAIN = BASE / 'train'
DST_TEST  = BASE / 'test'

# Create split dirs
for d in (DST_TRAIN, DST_TEST):
    d.mkdir(parents=True, exist_ok=True)

def save_dic(obj: dict, filepath: Path):
    with open(filepath, 'wb') as fo: 
        pickle.dump(obj, fo)


# Parameters
TEST_SIZE  = 0.2
RANDOM_SEED = 42

class_index = {}
# For each class:
i = 1
for class_dir in SRC.iterdir():
    if not class_dir.is_dir(): 
        continue
    class_index[class_dir.name] = str(i)
    i += 1
    if i > 21: 
        break
    
    images = list(class_dir.glob('*'))  # all files in that class
    train_imgs, test_imgs = train_test_split(
        images,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # ensure class subfolder exists
    (DST_TRAIN / class_index[class_dir.name]).mkdir(exist_ok=True)
    (DST_TEST  / class_index[class_dir.name]).mkdir(exist_ok=True)

    # copy training files
    for img_path in train_imgs:
        shutil.copy2(img_path, DST_TRAIN / class_index[class_dir.name] / img_path.name)

    # copy testing files
    for img_path in test_imgs:
        shutil.copy2(img_path, DST_TEST  / class_index[class_dir.name] / img_path.name)

save_dic(class_index, BASE / 'class_index')
print(class_index)