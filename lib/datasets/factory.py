# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.tabletop_object
import datasets.osd_object
import datasets.ocid_object
import numpy as np

# tabletop object dataset
for split in ['train', 'test', 'all']:
    name = 'tabletop_object_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.TableTopObject(split))

# OSD object dataset
for split in ['test']:
    name = 'osd_object_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.OSDObject(split))

# OCID object dataset
for split in ['test']:
    name = 'ocid_object_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.OCIDObject(split))

def get_dataset(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_datasets():
    """List all registered imdbs."""
    return __sets.keys()
