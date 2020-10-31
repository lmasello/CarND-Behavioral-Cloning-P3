# **Behavioral Cloning Project**
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_centre]: ./examples/image_centre.jpg "Image of the centre camera"
[image_flipped]: ./examples/image_flipped.jpg "Flipped Image"
[image_eq]: ./examples/image_eq.jpg "Eq Image"
[track_1]: ./examples/track_1.gif "Track 1"
[track_2]: ./examples/track_2.gif "Track 2"

![Autonomous driving on track 1][track_1]
![Autonomous driving on track 2][track_2]

## Rubric Points
List of [rubric points](https://review.udacity.com/#!/rubrics/432/view) to pass the project.  

---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to preprocess the images and train the model
* model.h5 containing a trained convolution neural network
* drive.py for driving the car in autonomous mode
* video.py to create a video recording from the output of drive.py
* track_1.mp4 containing the video of the autonomous driving on the first track
* track_2.mp4 containing the video of the autonomous driving on the second track
* writeup_report.md summarizing the results

### 2. Submission includes functional code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for augment the images, and training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the NVIDIA ["End to End Learning for Self-Driving Cars"](https://arxiv.org/pdf/1604.07316.pdf) publication with the following modifications:
- The input images have a 160 x 320 x 3 shape (RGB colour channel).
- After the normalization layer I applied a cropping layer to focus only on the pixels of the road (i.g., avoid having pixels for the sky or the bonnet).
- I added one dropout layer after the convolution layers to avoid overfitting.

Additionally, the network has been built using the Keras library.

### 2. Attempts to reduce overfitting in the model

The model has been trained using 3 EPOCHS and contains one dropout layer in order to reduce overfitting (model.py line 97).

The model was trained and validated on different data sets (track 1 and track 2) to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track in both scenarios.

### 3. Model parameter tuning

The parameters used for this model are:
- EPOCHS = 3
- BATCH_SIZE = 64
- DROPOUT_RATE = 0.75

As for the learning rate, the model used an Adam optimizer, so there was not need to tune the learning rate manually (model.py line 106).

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road using not only the first training track but also the second one containing harder turns and hills. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving counter clockwise. The next section describes details about how I created the training data.

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different architectures and then select the one that presented the lowest validation loss.

My first step was to use a convolution neural network model following the LeNet architecture, after observing outstanding results on the previous project, [the traffic sign classifier](https://github.com/lmasello/CarND-Traffic-Sign-Classifier-Project). However, with this architecture the car was not able to stay on the center of the lane in some areas of the road so I decided to try a more powerful architecture. It turned out that the convolution neural network based proposed by the NVIDIA in the ["End to End Learning for Self-Driving Cars"](https://arxiv.org/pdf/1604.07316.pdf) publication improved the performance significantly. Then, I added a dropout layer to avoid overfitting and the car achieved the target results in both tracks. To gauge how well the model was working, I split the image and steering angle data into a training (80%) and validation set (20%). As a result, the model presented a low mean squared error on both datasets.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle lost the centre of the lane, in particular, on the bridge. To improve the driving behavior in these cases, I recorded more data with the simulator in manual mode.

At the end of the process, the vehicle was able to drive autonomously around both tracks without leaving the road.

### 2. Final Model Architecture

The final model architecture (model.py lines 89-102) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		     |     Description	        					|
|:-----------------------|:---------------------------------------------:|
| Input         		     | 160x320x3 RGB image   					|
| Normalization (Lambda) | MinMax normalization centred around 0 |
| Cropping2D     	       | Crop 50px from the top and 20px from the bottom |
| Conv2D                 | 24 filters, 5x5 kernel, strides 2x2, VALID padding, relu activation |
| Conv2D                 | 36 filters, 5x5 kernel, strides 2x2, VALID padding, relu activation |
| Conv2D                 | 48 filters, 5x5 kernel, strides 2x2, VALID padding, relu activation |
| Conv2D                 | 64 filters, 3x3 kernel, strides 1x1, VALID padding, relu activation |
| Conv2D                 | 64 filters, 3x3 kernel, strides 1x1, VALID padding, relu activation |
| Dropout                |  |
| Flatten                |  |
| Fully-connected layer (Dense) | 100 units, relu activation |
| Fully-connected layer (Dense) | 50 units, relu activation |
| Fully-connected layer (Dense) | 10 units, relu activation |
| Fully-connected layer (Dense) | 1 unit |

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track 1 using center lane driving, using the images captured by the left, centre and right cameras. Here is an example image of center lane driving:

![Image from the central camera][image_centre]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover the course to the centre of the lane. The same process was repeated on two laps of track 2 in order to get more data points in a different context. Additionally, in the first track I also drove counter-clockwise to avoid the left turn bias inherent to the road.

To augment the dataset, I flipped the images and angles to simulate the vehicle in opposite scenarios, and also applied [histogram equalization](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html) to improve contrast. For instance, here is an image that has then been flipped:

![Image from the central camera][image_centre]
![Flipped image from the central camera][image_flipped]

Then, the same image with histogram equalization:

![Image with histogram equalization][image_eq]

The manual driving data collected was combined with the sample driving data, and as a result I had a total of 15,325 data points. I finally randomly shuffled the data set and put 20% of the data into a validation set.
