# Human Action Recognition for Still Images

The models trained using pytorch framework appear to be getting higher accuracy. 
Possible differences might be change in 

* batch_size
* framework
* models here do not include the extra Conv2d layer + Global Average Pooling used in the Keras models

... Turns out that I forgot to replace the last layer's 1000-way softmax with a 40-way softmax -- yet the models still did surprisingly well.  May get an additional boost in accuracy if I re-run with a 40-way softmax as the output layer.  

Additionally, I forgot to freeze the parameters of all other layers as well.  Since the models still performed well, actually even better than the Keras counterparts where the layers were frozen, it seems that the Stanford 40 action is large enough that no layers need to be frozen.  Should re-run also these parameters frozen as well to see how that affects accuracy.

## Results 

In the table below, pixels represents the spatial resolution of the input to a model.
Training was run with the input set to 224x224 RGB images as well as 500x500 RG images.

These are the results for the model that still uses a 1000-way softmax (all layers trainable)
The model still did surprisingly well.

  Model        |    Pixels  |   Accuracy
---------------|------------|--------------
VGG-16         |    224/500 |    79.4% / -
VGG-19         |    224/500 |    78.0% / -
MobileNet-v2   |    224/500 |    76.9% / -
ResNet-50      |    224/500 |    82.8% / 88.3%
ResNeXt-50     |    224/500 |    83.5% / 90.0%
ResNeXt-101    |    224/500 |    85.5% / 90.4%
DenseNet-121   |    224/500 |    82.3% / 88.4%

The results after replacing the final layer with a 40-way softmax (all layers trainable)
Need to re-run to fill in results.  

  Model        |    Pixels  |   Accuracy
---------------|------------|--------------
VGG-16         |    224/500 |    79.7% / 80.6%
VGG-19         |    224/500 |    80.9% / -
MobileNet-v2   |    224/500 |    79.4% / 83.9%
ResNet-50      |    224/500 |    83.8% / 89.0%
ResNeXt-50     |    224/500 |    85.7% / -
ResNeXt-101    |    224/500 |    - / 90.7%
DenseNet-121   |    224/500 |    - / 87.1%



## Training Logs

VGG16 training results:


<img src="images/vgg16_acc.jpg" width="45%" /><img src="images/vgg16_loss.jpg" width="45%"/>


VGG19 training results:

<img src="images/vgg19_acc_loss.png" />


MobileNet-V2 training results:

<img src="images/mobilenet-v2.png" />

ResNet-50 training results.  Validation accuracy: 82.8%, strangely a lot better than the version using Keras where the accuracy was as good as guessing.

<img src="images/resnet50_acc_loss.jpg" >
