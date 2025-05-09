from pathlib import Path
import numpy as np 
import cv2
import pickle

def unpickle(file):
    with open(file, 'rb') as fo: 
        dict = pickle.load(fo, encoding='latin1')
    return dict

def image(array): 
    r_flat = array[0:1024]
    g_flat = array[1024:2048]
    b_flat = array[2048:3072]

    r = r_flat.reshape(32, 32)
    g = g_flat.reshape(32, 32)
    b = b_flat.reshape(32, 32)

    img_array = np.stack([b, g, r], axis=2)
    return img_array

dataset_path = Path("/home/REDACTED/Projects/manchester/computer_vision/datasets/")
filepath = dataset_path / "raw/cifar-10-batches-py"
    

meta_file = unpickle(dataset_path / "raw/cifar-10-batches-py/batches.meta")

labelling_dic = {id + 1: name for id, name in enumerate(meta_file['label_names'])}
print(labelling_dic)

output_dir = Path(
    "/home/REDACTED/Projects/manchester/computer_vision/datasets/processed/cifar-10/"
)

if not output_dir.is_dir():
    output_dir.mkdir()

meta_out = output_dir / "class_index"

with open(meta_out, 'wb') as fo:
    pickle.dump(labelling_dic, fo)

for file in list(filepath.glob('*_batch*')):
    data = unpickle(file)
    batch_label = data['batch_label']
    print(data['batch_label'])
    if 'testing' in batch_label:
        subset_dir = output_dir / 'test'
    else:
        subset_dir = output_dir / 'train'

    if not subset_dir.is_dir():
        subset_dir.mkdir()

    for i, array in enumerate(data['data']):
        img = image(array)
        lbl = data['labels'][i]
        filename = data['filenames'][i]
        
        lbl_dir = subset_dir / str(lbl + 1) 
    
        if not lbl_dir.is_dir():
            lbl_dir.mkdir()
    
        output_img = lbl_dir / filename
        cv2.imwrite(output_img, img) 
