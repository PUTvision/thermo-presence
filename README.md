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
Our dataset is publicly available. Additionally, we provide a notebook and a trained model that allow you to replicate the experiments. You can find everything [here](https://chmura.put.poznan.pl/s/MftZtediKJOrL3f).

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
