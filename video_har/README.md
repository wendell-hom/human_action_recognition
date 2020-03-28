# Human Action Recognition for Videos

For Human Action Recognition with Videos, three different models were evaluated.
The first model uses a 2D CNN we fine-tuned from the image action recognition protion of this project.
The second model is the C3D architecture, a shallow 3D CNN that was mentioned in the paper by Tran et-el.
The third model is a 3D version of the ResNeXt-101, a deep 3D CNN that was used in the paper by Hara et-el.


|               |  Adam Opt     |  SGD Opt      | 
|---------------|---------------| --------------|
| 2D CNN        |    59.0%      | -             |
| C3D           |    35.2%      | -             |
| ResNeX-t101   |    24.4%      | 28.1%         |

