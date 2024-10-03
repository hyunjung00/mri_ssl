# Brain Tumor Segmentation 

# Introduction 

The objective of this project is to develop an efficient and accurate deep learning-based approach for brain tumor segmentation using the Swin UNETR model.The motivation behind this project is to contribute to the advancement of automated medical image analysis, which can aid clinicians in the early diagnosis and treatment planning of brain tumors. Accurate segmentation of brain tumors from MRI scans is critical for treatment planning, assessment of disease progression, and overall patient management. However, the heterogeneity of tumors, varying sizes, shapes, and intensities make this a challenging task. The Swin UNETR model combines the Swin Transformer backbone with a UNET architecture, making it highly effective for extracting both local and global features from 3D medical images.

# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

# Data Description

Modality: MRI
Size: 1470 3D volumes (1251 Training + 219 Validation) / Test dataset 53 MRI 
Challenge: RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge

- Register and download the official BraTS 21 dataset from the link below and place them into "TrainingData" in the dataset folder:

  https://www.synapse.org/#!Synapse:syn27046444/wiki/616992

  For example, the address of a single file is as follows:

  "TrainingData/BraTS2021_01146/BraTS2021_01146_flair.nii.gz"


- Download the json file from this [link](https://drive.google.com/file/d/1i-BXYe-wZ8R9Vp3GXoajGyqaJ65Jybg1/view?usp=sharing) and placed in the same folder as the dataset.


The sub-regions considered for evaluation in BraTS 21 challenge are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT). The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (NCR) parts of the tumor. The appearance of NCR is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edematous/invaded tissue (ED), which is typically depicted by hyper-intense signal in FLAIR [[BraTS 21]](http://braintumorsegmentation.org/).

The provided segmentation labels have values of 1 for NCR, 2 for ED, 4 for ET, and 0 for everything else.

# Models
Swin UNETR models which are pre-trained on BraTS21 dataset as in the following. The folds
correspond to the data split in the [json file](https://drive.google.com/file/d/1i-BXYe-wZ8R9Vp3GXoajGyqaJ65Jybg1/view?usp=sharing).

<table>
  <tr>
    <th>Name</th>
    <th>Fold</th>
    <th>Mean Dice</th>
    <th>Feature Size</th>
    <th># params (M)</th>
    <th>Download </th>
  </tr>
<tr>
    <td>Swin UNETR</td>
    <td>0</td>
    <td>88.54</td>
    <td>48</td>
    <td>62.1</td>
    <td><a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold0_f48_ep300_4gpu_dice0_8854.zip">model</a></td>
</tr>

<tr>
    <td>Swin UNETR</td>
    <td>1</td>
    <td>90.59</td>
    <td>48</td>
    <td>62.1</td>
    <td><a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold1_f48_ep300_4gpu_dice0_9059.zip">model</a></td>
</tr>

<tr>
    <td>Swin UNETR</td>
    <td>2</td>
    <td>89.81</td>
    <td>48</td>
    <td>62.1</td>
    <td><a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold2_f48_ep300_4gpu_dice0_8981.zip">model</a></td>
</tr>

<tr>
    <td>Swin UNETR</td>
    <td>3</td>
    <td>89.24</td>
    <td>48</td>
    <td>62.1</td>
    <td><a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold3_f48_ep300_4gpu_dice0_8924.zip">model</a></td>
</tr>

<tr>
    <td>Swin UNETR</td>
    <td>4</td>
    <td>90.35</td>
    <td>48</td>
    <td>62.1</td>
    <td><a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold4_f48_ep300_4gpu_dice0_9035.zip">model</a></td>
</tr>

</table>

Mean Dice refers to average Dice of WT, ET and TC tumor semantic classes.

# Training

A Swin UNETR network with standard hyper-parameters for brain tumor semantic segmentation (BraTS dataset) is be defined as:

``` bash
model = SwinUNETR(img_size=(128,128,128),
                  in_channels=4,
                  out_channels=3,
                  feature_size=48,
                  use_checkpoint=True,
                  )
```

