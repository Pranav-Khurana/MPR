import numpy as np 
import pandas as pd
import nilearn.image as image
from dyneusr import DyNeuGraph 
from nilearn.plotting import plot_roi, plot_epi, show
from nilearn.datasets import  fetch_miyawaki2008
from nilearn.input_data import NiftiMasker

from kmapper import KeplerMapper, Cover
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN


# Fetch dataset, extract time-series from ventral temporal (VT) mask
dataset = fetch_miyawaki2008()

# Fetch dataset, extract time-series from ventral temporal (VT) mask
dataset = fetch_miyawaki2008()
masker = NiftiMasker(
    dataset.mask_vt[0], 
    standardize=True, detrend=True, smoothing_fwhm=4.0,
    low_pass=0.09, high_pass=0.008, t_r=2.5,
    memory="nilearn_cache")
X = masker.fit_transform(dataset.func[0])