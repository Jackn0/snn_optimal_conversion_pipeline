# snn_optimal_conversion_pipeline
 Optimal Conversion of Conventional Artificial Neural Networks to Spiking Neural Networks

# Training and simulation
We suggest using file 'main_train.py' for training and file 'main_simulation.py' for simulation. 
* The training and simulation parameters are collected in 'models/settings.py'.

# Files
* 'main_train.py' : main training file.
* 'main_simulation.py' : main simulation file.
* 'models/settings.py' : collection of the parameters.
* 'models/spiking_layer.py' : SPIKE_layer to replace ANN's convolution layer and linear layer.
* 'models/new_relu.py' : threshold ReLU file

# Pre-trained models
* All the pre-trained models we used are avilabled [here](https://drive.google.com/drive/folders/1JAAtdOTcmbfv732aRlqsdL7-_05Tdemu?usp=sharing)

# Issues
* Consider the generalization, when T is large, the loss and accuracy of SNN may both decrease.

# Citation
If our code is helpful to you, please cite the following [paper](https://openreview.net/forum?id=FZ1oTwcXchK).
```
@inproceedings{
deng2021optimal,
title={Optimal Conversion of Conventional Artificial Neural Networks to Spiking Neural Networks},
author={Shikuang Deng and Shi Gu},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=FZ1oTwcXchK}
}
```

