# Copyright (C) 2022-present Naver Corporation / Inria centre at the University Grenoble Alpes. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import glob
import argparse
from ipdb import set_trace as bb
osp = os.path
import numpy as np

class SHOWMeDatasetInterface(object):
    """
    Basic dataset class for showme dataset, kind of interface to dataset
    """
    def __init__(self, datadir=None):
        self.datadir = datadir

    def __len__(self):
        """
        returns the number of sequences
        """
        return len(self.get_all_sqns_ids())

    def get_all_sqns_ids(self,):
        all_sqns_ids = os.listdir(self.datadir)
        return all_sqns_ids
    
    def get_seq_rgbs_pths(self, sqn):
        "get seq's rgbs pths (note: no of rgb files could be diff from no of pose anno files)"
        seq_rgbs_pths = sorted(glob.glob(osp.join(self.datadir, sqn, 'rgb/*.png')))
        return seq_rgbs_pths
    
    def get_seq_pose_pths(self, sqn):
        "get seq's pose anno pths (note: no of pose anno files could be diff from no of rgb files)"
        seq_poses_pths = sorted(glob.glob(osp.join(self.datadir, sqn, 'icp_res/*/*.txt')))[1:]
        return seq_poses_pths

    def __str__(self,):
        """
        print the info of a one sample of data
        """
        return "\n".join([
            f"{type(self).__name__} (#seqs_ids = {len(self)},  dataset root dir : {self.datadir})"
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='dataset root dir pth') 
    args = parser.parse_args()

    dset_interf = SHOWMeDatasetInterface(datadir=args.datadir)
    print('\n', str(dset_interf), '\n')
    print(f"Sequence ids: \n", dset_interf.get_all_sqns_ids())


