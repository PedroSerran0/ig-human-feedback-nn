# Imports
import os
from typing_extensions import assert_type
import _pickle as CPickle
import numpy as np



# ROSE Database
# Function: Get images of a specific split and class
def get_images(data_split, img_class, attack_type, data_path='data/ROSE_DB/data_divided'):

    # Assert conditions
    # Data split
    assert data_split in ('train', 'test'), f"Data split should be 'train' or 'test. You entered {data_split}."

    # Image class
    assert img_class in (0, 1), f"Image class should be either 0 (bonafide) or 1 (attack). You entered {img_class}."



    # Enter data_split directory
    data_split_dir = os.path.join(data_path, data_split)

    # Enter img_class directory
    img_class_dir = os.path.join(data_split_dir, img_class)


    # If img_class is 0, we return all the images there
    if img_class == 0:
        images = [i for i in os.listdir(img_class_dir) if not i.startswith('.')]
    
    else:
        # TODO: Review
        if attack_type not None:
            assert attack_type in [i for i in range(1, 8)], f"Attack type should be an integer between 1-7. You entered {attack_type}."
            # TODO: Add this
        
        else:
            images = [i for i in os.listdir(img_class_dir) if not i.startswith('.')]


    return images
