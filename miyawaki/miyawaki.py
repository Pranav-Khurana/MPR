import numpy as np 
import pandas as pd
from dyneusr import DyNeuGraph 

from nilearn.datasets import  fetch_miyawaki2008
from nilearn.input_data import MultiNiftiMasker

from kmapper import KeplerMapper, Cover
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# Fetch dataset, extract time-series from ventral temporal (VT) mask
dataset = fetch_miyawaki2008()
masker = MultiNiftiMasker(mask_img=dataset.mask, detrend=True,
                          standardize=False)
X = masker.fit_transform(dataset.func[0])

# Encode labels as integers
#df = pd.read_csv(dataset.session_target[0], sep=" ")
#target, labels = pd.factorize(df.labels.values)
#y = pd.DataFrame({l:(target==i).astype(int) for i,l in enumerate(labels)})

# Generate shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=TSNE(2))
graph = mapper.map(lens, X, cover=Cover(20, 0.5), clusterer=DBSCAN(eps=20.))

# Visualize the shape graph using DyNeuSR's DyNeuGraph                          
dG = DyNeuGraph(G=graph)
dG.visualize('output.html')

