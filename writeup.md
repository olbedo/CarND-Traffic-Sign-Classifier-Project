# Traffic Sign Recognition

## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[examples_per_class]: ./pics/examples_per_class.png "Number of instances per class"
[train_16266]: ./pics/train_16266(8).png "Instance 16266 of the training set"
[train_16266_equal]: ./pics/train_16266(8)_equal.png "Instance 16266 after equalization"
[cnn_graph]: ./pics/cnn_graph.png "Final Model Architecture"
[image1]: ./pics/sign1.jpg "Traffic Sign 1"
[image2]: ./pics/sign2.jpg "Traffic Sign 2"
[image3]: ./pics/sign3.jpg "Traffic Sign 3"
[image4]: ./pics/sign4.jpg "Traffic Sign 4"
[image5]: ./pics/sign5.jpg "Traffic Sign 5"
[feature_maps_conv1]: ./pics/feature_maps_conv1.png "Future maps of the first convolutional layer"
[feature_maps_conv2]: ./pics/feature_maps_conv2.png "Future maps of the second convolutional layer"

### Data Set Summary & Exploration


I used numpy methods to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread over the individual classes/labels. As one can see, the examples are not evenly distributed over the classes. Some classes have only 180 examples whereas for other classes there are more than 2000 instances. This is a factor of more than 10. It is possible, that this circumstance has a negative impact on the performance of the classifier.

![examples_per_class][examples_per_class]

Figure 1: Number of instances per class

While investigating the data, I realized that some images are very dark as shown in Figure 2. Therefore the first thing to do in the preprocessing step would be to equalize the images.

![train_16266][train_16266]

Figure 2: Instance 16266 of the training set

### Design and Test a Model Architecture

#### Preprocessing

As mentioned above, I decided to equalize the images first. For this task I used skimage. `exposure.equalize_adapthist` as it turn out to bring the best results. The following figure shows the image from Figure 2 after.

![train_16266_equal][train_16266_equal]

Figure 3: Instance 16266 after equalization

I decided not to convert the images to grayscale because in my tests it led to slightly worse results.
As a last step, I normalized the image data as this is necessary for the ConvNet to work properly. I used the formula

X = (X-X_mean)/X_mean

rather than whitening the images (considering also the standard deviation) since it produced better results.
So far, I did not generate additional data. This will be the next step to further improve the test accuracy.
Model Architecture
My final model consisted of the following layers:

| Layer           | Description	        					              |
|:---------------:|:-------------------------------------------:|
| Input           |	32x32x3 RGB image                           |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6  |
| ELU	            |	worked better than ReLU                     |
| Max pooling     |	2x2 stride, valid padding, outputs 14x14x6  |
| Dropout         | keep probability 0.8                        |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
| ELU	            |	worked better than ReLU                     |
| Max pooling     |	2x2 stride, valid padding, outputs 5x5x16   |
| Dropout         | keep probability 0.8                        |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 1x1x240  |
| ELU	            |	worked better than ReLU                     |
| Dropout         | keep probability 0.8                        |
| Flatten         | outputs 240                                 |
| Fully connected	| outputs 170                                 |
| Dropout         | keep probability 0.8                        |
| Softmax         | outputs logits (43)                         |

The dropout layers were of course only activated during training.

Figure 4 depicts the final model architecture created by `tensorboard`.

#### Training
To train the model, I used `AdamOptimizer` as it seems to be the best choice at the moment. The weights in the model were initialized acc. to Xavier. Furthermore I applied Batch Normalization with a decay rate of 0.99.
I have chosen a batch size of 128, number of epochs of 10 and a learning rate of 0.001 as these hyperparameters led to the best results in the final model. I did not use learning rate decay and L2 normalization since it decreased the validation accuracy.

#### Solution Finding
As recommended in the project description, I started with the model from LeNet-Lab and got about 88% test accuracy. Then I tried to improve the performance by equalizing the image data before feeding it into the CNN. This raised the accuracy about 2 percentage points. I tried to convert the date to grayscale but it did not make much difference, in fact in decreased the performance slightly. After introducing dropout into the ConvNet and He-initialization of the weights I reached almost 93% test accuracy.

![cnn_graph][cnn_graph]

Figure 4: Final Model Architecture

I played around with the model by adding more layers, changing the number of feature maps, kernel sizes (also 1x1 convolutions) and padding methods. Some changes I did just to see the impact of them and get a better feeling for the parameters of a CNN. Other changes were inspired by literature about the topic, e.g. by Yann LeCun. But in most cases the model was overfitting. To prevent this, I tested different values for the keep probability for dropout, applied L2 normalization and tried different values for the learning rate including introduction of learning rate decay. But none of them really helped.

Only when I applied batch normalization the validation accuracy increased faster and converged at a higher level. I tried to tweak the model by changing the batch size and the number of epochs. But it did not have much impact on the accuracy. Also, whitening did not bring any further improvement. On the other hand, using ELUs as activation function rather than ReLUs resulted in a slightly better test accuracy. Finally, I changed from He initialization to Xavier initialization as it brings similar results and it is already implemented in TensorFlow (so I guess it is a bit faster than my handwritten implementation).

My final model results were:
*	training set accuracy of 99.4%
*	validation set accuracy of 97.5%
*	test set accuracy of 96.2

### Test a Model on New Images
#### Selection of Images
Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3]
![alt text][image4] ![alt text][image5]

Figure 5: German traffic signs found on the web

The first image might be difficult to classify because it is not exactly the one which was trained, it is smaller than the signs in the test set and it is a bit tilted. The other images might be difficult to classify because they are bigger than the ones in the training set (especially the second one). Also, the sign in the second image is scratched and the last sign is again tilted. But in general, I expect a good classification performance.

#### Discussion of Predictions

Here are the results of the prediction:

| Image			           |     Prediction	       |
|:--------------------:|:---------------------:|
| Speed limit (30km/h) | Speed limit (30km/h)  |
| Speed limit (50km/h) | Speed limit (50km/h)  |
| Priority road	       | Priority road         |
| Stop	               | Stop                  |
| No entry	           | No entry              |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.2%. Of course the values are not the same, since the number of images is much smaller than in the test set.

#### Softmax Probabilities
The code for making predictions on my final model is located in the 22nd cell of the Ipython notebook.

For the first image, the model is not particularly sure that this is a 30km/h speed limit sign (probability of 0.25), but nonetheless the image does contain in fact this sign and the probability of the second best prediction is much lower.

The top five soft max probabilities were

| Probability	| Prediction	        		        			|
|:-----------:|:-------------------------------------:|
| 0.25        | Speed limit (30km/h)                  |
| 0.09        | Priority road                         |
| 0.08        | Roundabout mandatory                  |
| 0.05        | Speed limit (50km/h)                  |
| 0.04        | Right-of-way at the next intersection |

For the second image, the model is even less sure. Even though it classifies the image correctly, the first and the second guess have almost the same probability. This is understandable as the 50 km/h and the 30km/h speed limit signs are not easily to distinguish at such low resolution. The top five soft max probabilities were

| Probability	| Prediction	        		        			       |
|:-----------:|:--------------------------------------------:|
| 0.21        | Speed limit (50km/h)                         |
| 0.21        | Speed limit (30km/h)                         |
| 0.07        | No passing for vehicles over 3.5 metric tons |
| 0.05        | Priority road                                |
| 0.03        | Keep right                                   |

In the third to fifth image, the model is relatively sure about its prediction. The probabilities are higher than 70% for the first guess and less than 3% for all other guesses. The top five soft max probabilities of the third image were

| Probability	| Prediction	        		        			       |
|:-----------:|:--------------------------------------------:|
| 0.86        | Priority road                                |
| 0.01        | Speed limit (50km/h)                         |
| 0.01        | Speed limit (30km/h)                         |
| 0.01        | No passing for vehicles over 3.5 metric tons |
| 0.01        | Keep right                                   |

The top five soft max probabilities of the forth image were

| Probability	| Prediction	         |
|:-----------:|:--------------------:|
| 0.82        | Stop                 |
| 0.01        | Speed limit (80km/h) |
| 0.01        | Speed limit (50km/h) |
| 0.01        | Road work            |
| 0.01        | Speed limit (30km/h) |

And the top five soft max probabilities of the fifth image were

| Probability	| Prediction	         |
|:-----------:|:--------------------:|
| 0.74        | No entry             |
| 0.02        | Speed limit (50km/h) |
| 0.02        | Speed limit (80km/h) |
| 0.02        | No passing           |
| 0.02        | Speed limit (30km/h) |

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

The following figure shows the output of the feature maps of the first convolutional layer after feeding the model with image 1. The outlines of the signs are more or less visible. Each feature map focuses on a specific aspect in the image, i.e. simple features like horizontal, vertical and diagonal lines. The activations slightly overlap. This redundancy is probably also caused by the dropout step.

![feature_maps_conv1][feature_maps_conv1]

Figure 6: Future maps of the first convolutional layer

After the Max-Pooling step, which follows the first convolutional layer, the information about the features is condensed in fewer cells. These are fed to the second convolutional layer with a much deeper feature depth. Now the individual features which activate the respective cells are not so clearly visible by human eyes anymore (see Figure 7). The simple features from layer 1 are combined to higher level features which cause the cells in the higher layers to fire when such features are recognized.

![feature_maps_conv2][feature_maps_conv2]

Figure 7: Future maps of the first convolutional layer
