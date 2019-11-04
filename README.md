# Understanding the Concepts of Brain Nueroimaging 

The Primary objective of this project is to understand and analyze various experiments conducted on brain fMRI images so as provide a deep understanding of human brain. This helps us in understanding, the working of our brain and also helps in detecting and providing solutions to health problems related to brain such as brain tumour.

### Experimental Analysis

- [*] [Haxby et al. (2001)](#Haxby_Experiment)
- [ ] [Miyawaki et al. (2008)](#miyawaki)

---

## Haxby_Experiment

In the original work, visual stimuli from 8 different categories are presented to 6 subjects during 12 sessions. The goal is to predict the category of the stimulus presented to the subject given the recorded fMRI volumes. For the sake of simplicity, we restrict the example to one subject and try to analyse the stimulus as per the presented images and in resting state as well.

```python
import numpy as np 
import pandas as pd
from dyneusr import DyNeuGraph 
from nilearn.datasets import fetch_haxby
from nilearn.input_data import NiftiMasker
from kmapper import KeplerMapper, Cover
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# Fetch dataset, extract time-series from ventral temporal (VT) mask
dataset = fetch_haxby()
masker = NiftiMasker(
    dataset.mask_vt[0], 
    standardize=True, detrend=True, smoothing_fwhm=4.0,
    low_pass=0.09, high_pass=0.008, t_r=2.5,
    memory="nilearn_cache")
X = masker.fit_transform(dataset.func[0])

# Encode labels as integers
df = pd.read_csv(dataset.session_target[0], sep=" ")
target, labels = pd.factorize(df.labels.values)
y = pd.DataFrame({l:(target==i).astype(int) for i,l in enumerate(labels)})

# Generate shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=TSNE(2))
graph = mapper.map(lens, X, cover=Cover(20, 0.5), clusterer=DBSCAN(eps=20.))

# Visualize the shape graph using DyNeuSR's DyNeuGraph                          
dG = DyNeuGraph(G=graph, y=y)
dG.visualize('haxby/haxby_output.html')
```

The results of the same can be observed over here [Haxby_output](haxby/haxby_output.html)

---
