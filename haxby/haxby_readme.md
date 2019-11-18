# Haxby Experiment

In the original work, visual stimuli from 8 different categories are presented to 6 subjects during 12 sessions. The goal is to predict the category of the stimulus presented to the subject given the recorded fMRI volumes. For the sake of simplicity, we restrict the example to one subject and try to analyse the stimulus as per the presented images and in resting state as well.

```python
#Importing required libraries and dataset
import matplotlib.pyplot as plt
from nilearn import image
from nilearn import datasets
from nilearn.plotting import plot_anat, show
from matplotlib.patches import Rectangle

#Fetching the Dataset
haxby_dataset = datasets.fetch_haxby()

#Building the mean image because we have no anatomic data
func = haxby_dataset.func[0]
mean_img = image.mean_img(func)

z_slice = -14

fig = plt.figure(figsize=(4, 6.4), facecolor='k')(https://raw.githubusercontent.com/Pranav-Khurana/MPR/master/haxby/

display = plot_anat(mean_img, display_mode='z', cut_coords=[z_slice], figure=fig)

#Plotting the Brain Activities for ventral visual/temporal cortex, House and Face
vt = haxby_dataset.mask_vt[0]
house = haxby_dataset.mask_house[0]
face = haxby_dataset.mask_face[0]

display.add_contours(vt, contours=1, antialiased=False, linewidths=2., levels=[0], colors=['red'])
display.add_contours(house, contours=1, antialiased=False, linewidths=2., levels=[0], colors=['blue'])
display.add_contours(face, contours=1, antialiased=False, linewidths=2., levels=[0], colors=['yellow'])

#Generating Legends
p_v = Rectangle((0, 0), 1, 1, fc="red")
p_h = Rectangle((0, 0), 1, 1, fc="blue")
p_f = Rectangle((0, 0), 1, 1, fc="yellow")
plt.legend([p_v, p_h, p_f], ["ventral temporal", "house", "face"])

show()
```

[VIEW JUPYTER NOTEBOOK FILE](https://raw.githubusercontent.com/Pranav-Khurana/MPR/master/haxby/haxby.ipynb)


![OUTPUT](https://raw.githubusercontent.com/Pranav-Khurana/MPR/master/haxby/haxby_output.png)
