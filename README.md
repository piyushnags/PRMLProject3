# PRML Project 3: Deep Learning
Github Repository for PRML Project 3.
Look at demo.ipynb to reproduce all results presented in the report.
A GPU runtime is ideal for reproducing results, see notebook for details.

Best Classification Accuracy on Challenge Set: 62% using ResNet101
Optimal Solution: 42% Test Accuracy using MobileNetv3

classification.py: 
  - Contains main scripts for training MLP and custom CNN models. 
  - Supports transfer learning for pretrained models (Extra Credit)
  - NOTE: Checkpointing method is a bit hacky

model.py:
  - Contains all models

## Quickstart 
After installing the required packages, you can run the code to train each "variant" of the datasets with (assuming your system is compatible with the default arguments):

```
python classification.py --dataset Taiji --train --fp_size full
python classification.py --dataset Taiji --train --fp_size lod4
python classification.py --dataset Wallpaper --train --test_set test
python classification.py --dataset Wallpaper --train --test_set test_challenge
```


## Data
- data/taiji_data_full.npz (same as P2)
  - The data you will be working with
  - Variables (np arrays):
    - **feature_names**: the names of all 1961 features
    - **form_names**: the names of all 45 forms you will be trying to classify
    - **labels**: provided class labels corresponding to each data point
    - **sub_info**: information about subject number and performance number corresponding to each data point
    - **data**: the actual data you will be working with
- data/taiji_data_lod4_fp.npz (same as taiji_data_full except with downsampled foot pressure)

- data/wallpapes
    - tain and test directories containing the images for the wallpaper dataset
    - Both train and test have directories for each class containing the images for that class.
    - test_augment contains the augmented test images (harder to classify)

