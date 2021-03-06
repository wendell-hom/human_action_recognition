# Still Image Action Recognition using Keras Framework

Image action recognition was initially implemented using the Keras framework, and currently contains a more complete 
account of all models and their results.  On the other hand, the models that were fine-tuned using the pytorch framework currently have accuracy rates that are higher than their Keras counterparts for the same input resolution.


## Results

Below is the accuracy on the validation set of a number of models that were fine-tuned using the Stanford40 dataset

<img src="images/image_results_table.png">

## Confusion Matrix
Below is the confusion matrix for two different models, ResNet-50 on the left, and Inception-ResNet-v2 on the right.
The results for the ResNet-50 was terrible, as you can see from the confusion matrix it was simply picking one category for all images it did not see in the training set.  On the other hand, the ResNet-50 model from the pytorch version performed very well.  This could be due to a number of factors including better augmentation, different batch size, and differences in the models or trained weights between the Keras vs. torchvision model.

<img src="images/ResNet-50_cmat.png" width="45%"/><img src="images/Inception_ResNet_v2.png" width="45%" />

## Class Activity Mapping

The class activation map is shown below for a few images that were run on the InceptionResNet-V2 model.

<img src="images/brushing.png" /> <img src="images/jumping.png" /> <img src="images/violin1.png" /> 
