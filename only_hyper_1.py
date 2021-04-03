# -*- coding: utf-8 -*-

"""
Created on Thu Apr  1 11:19:17 2021

@author: hlw69
"""


####### THIS WORKS, FIND OUT EXACTLY WHICH SCRIPTS IT IS CALLING AND SAVE TO GITHUB 
##### MAKE CHANGE IN NITT from decomp svd to linalg non gpu svd!!
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from nibabel import save
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.labs.viz import plot_map, cm
from nipy.modalities.fmri.experimental_paradigm import  BlockParadigm # EventRelatedParadigm
from os import mkdir, path
from itertools import chain
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import sklearn
import glob
import nibabel
import os
#from nltools.stats import align
from stats_hw import align
import matplotlib.pyplot as plt
import warnings
import json
#from mvpa2.suite import *
import nibabel as nib
from nilearn.masking import apply_mask as nimask

with open('dicts/dict_faces_perception_roi_hex.json') as fp:
    dict_faces_perception = json.load(fp)

print("Data is laoded")

warnings.simplefilter('ignore')
all_data = []
for subject in range(30,59):
    if subject == 39:
        print("skip 39")
    else:
        print(subject)
        data = np.array(dict_faces_perception[str(subject)])
        data = data.astype(np.float32)
        scaled = sklearn.preprocessing.minmax_scale(data, feature_range=(0, 1), axis=0, copy=True)
        print(np.min(scaled), np.max(scaled))
        all_data.append(scaled)
        
        
       
        
#lst = [item[:5,:100] for item in all_data[:3]] ## this works
lst = [item[:5,:100] for item in all_data[:3]] 
print(len(lst))        
print("data i processed into right format, wit h scaling, will begin hyperalignment")
hyperalign = align(lst, method = 'procrustes')
print("hyperalignment is complete")

print(hyperalign['transformation_matrix'])
