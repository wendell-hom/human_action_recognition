# Human Action Recognition for Videos

For Human Action Recognition with Videos, three different models were evaluated.
The first model uses a 2D CNN we fine-tuned from the image action recognition protion of this project.
The second model is the C3D architecture, a shallow 3D CNN that was mentioned in the paper by Tran et-el.
The third model is a 3D version of the ResNeXt-101, a deep 3D CNN that was used in the paper by Hara et-el.


## Results of Validation Set

These runs were done using Adam Optimizer

|               |  15-category  |  27-category  |    600-category
|---------------|---------------| --------------|-----------------
| 2D CNN        |    59.0%      |     -         |       -
| C3D           |    42.5%      |     43.7%     |      35.2%
| ResNeX-t101   |    41.1%      |     37.6%     |      24.4%


## SGD Optimizer

Training using the SGD Optimizer with lr = 0.1, decay = 0.001, and momentum = 0.9 was also launched for the ResNeXt-101 architecture on the full Kinetics-600 dataset.

Accuracy is currently at 28.1%
