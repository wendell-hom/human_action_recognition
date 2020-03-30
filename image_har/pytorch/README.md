# Human Action Recognition for Still Images

The models trained using pytorch framework appear to be getting higher accuracy. 
Possible differences might be change in 

* batch_size
* framework
* models here do not include the extra Conv2d layer + Global Average Pooling used in the Keras models

## Results

* ResNet-50 acc 82.8% @ 224x224x3
* ResNet-50 acc 88.3% @ 500x500x3


## Training Logs

VGG16 training results:


<img src="images/vgg16_acc.jpg" width="45%" /><img src="images/vgg16_loss.jpg" width="45%"/>


VGG19 training results:

<img src="images/vgg19_acc_loss.png" />


MobileNet-V2 training results:

<img src="images/mobilenet-v2.png" />

ResNet-50 training results.  Validation accuracy: 82.8%, strangely a lot better than the version using Keras where the accuracy was as good as guessing.

<img src="images/resnet50_acc_loss.jpg" >
