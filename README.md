# Thermo presence

<!-- ![main](https://github.com/PUTvision/thermo-presence/actions/workflows/ci.yml/badge.svg) -->
[![GitHub contributors](https://img.shields.io/github/contributors/PUTvision/thermo-presence)](https://github.com/PUTvision/thermo-presence/graphs/contributors)
[![GitHub stars](https://img.shields.io/github/stars/PUTvision/thermo-presence)](https://github.com/PUTvision/thermo-presence/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/PUTvision/thermo-presence)](https://github.com/PUTvision/thermo-presence/network/members)

## **Detect and count people on infrared images from a low resolution thermovision camera (24x32 pixels).**

> This is the official repo for the paper: [Low-Cost Thermal Camera-Based Counting Occupancy Meter Facilitating Energy Saving in Smart Buildings](https://www.mdpi.com/1996-1073/14/15/4542/htm)

## Overview
<p align="center">
    <img src="./README/plot.gif" height="200px" />
</p>

> Using passive infrared sensors is a well-established technique of presence monitoring. While it can significantly reduce energy consumption, more savings can be made when utilising more modern sensor solutions coupled with machine learning algorithms. This paper proposes an improved method of presence monitoring, which can accurately derive the number of people in the area supervised with a low-cost and low-energy thermal imaging sensor. The method utilises U-Net-like convolutional neural network architecture and has a low parameter count, and therefore can be used in embedded scenarios. Instead of providing simple, binary information, it learns to estimate the occupancy density function with the person count and approximate location, allowing the system to become considerably more flexible. The tests show that the method compares favourably to the state of the art solutions, achieving significantly better results.

## Table of contents
* [Dataset](#dataset)
* [Data collection](#data-collection)
* [Data processing](#data-processing)
* [Citation](#citation)

## Dataset
Our dataset is publicly available in the [dataset](./dataset/) directory which consists of data and labels folders. Recorded thermal images are in [data](./dataset/data) directory and corresponding annotations can be found in [labels](./dataset/labels) folder. Collected frames are divided into sequences which then are divided into training, validation, and test sets as follows:

```
training_dirs: [
    "006__11_44_59", "007__11_48_59", "008__11_52_59", "009__11_57_00", "000__14_15_19", "001__14_19_19", 
    "002__14_23_19", "003__14_27_20", "004__14_31_20", "012__15_03_21", "013__15_07_21", "014__15_11_21", 
    "015__15_15_21", "016__15_19_21", "011__13_38_20", "012__13_42_20", "013__13_46_21", "007__13_22_20"
]

validation_dirs: [
    "004__13_10_20", "014__13_50_21", "005__14_35_20", "006__14_39_20", "007__14_43_20", "008__14_47_20"
]

test_dirs: [
    "008__13_26_20", "009__14_51_20", "010__14_55_20", "011__14_59_20", "015__13_54_21"
]
```

The table below shows the summary of training, validation, and test datasets considering the number of people in the frame.

<div align="center">

|            |  0  |  1  |   2  |   3  |   4  |  5  | Total |
|:----------:|:---:|:---:|:----:|:----:|:----:|:---:|:-----:|
|  Training  |  99 | 105 | 2984 | 3217 | 1953 | 114 |  8472 |
| Validation |  0  | 139 |  631 | 1691 |  225 | 139 |  2825 |
|    Test    | 162 |  83 |  211 |  341 | 1235 | 314 |  2346 |

</div>

## Data collection
Software to record the data from camera.  
To be deployed on the RaspberryPi device.  
See [data_collection/README.md](./data_collection/README.md)

## Data processing
Software to process/analyse the data.  
See [data_processing/README.md](./data_processing/README.md)


## Citation

```
@Article{en14154542,
    AUTHOR = {Kraft, Marek and Aszkowski, Przemysław and Pieczyński, Dominik and Fularz, Michał},
    TITLE = {Low-Cost Thermal Camera-Based Counting Occupancy Meter Facilitating Energy Saving in Smart Buildings},
    JOURNAL = {Energies},
    VOLUME = {14},
    YEAR = {2021},
    NUMBER = {15},
    ARTICLE-NUMBER = {4542},
    URL = {https://www.mdpi.com/1996-1073/14/15/4542},
    ISSN = {1996-1073},
    DOI = {10.3390/en14154542}
}
```
