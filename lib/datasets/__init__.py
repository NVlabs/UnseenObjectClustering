# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

from .imdb import imdb
from .tabletop_object import TableTopObject
from .osd_object import OSDObject
from .ocid_object import OCIDObject

import os.path as osp
ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')
