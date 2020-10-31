# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


[//]: # (Image References)

[image_centre]: ./examples/image_centre.jpg "Image of the centre camera"
[image_flipped]: ./examples/image_flipped.jpg "Flipped Image"
[image_eq]: ./examples/image_eq.jpg "Eq Image"
[track_1]: ./examples/track_1.gif "Track 1"
[track_2]: ./examples/track_2.gif "Track 2"

![Autonomous driving on track 1][track_1]
![Autonomous driving on track 2][track_2]

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.
The simulator can be downloaded from the classroom.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

## Rubric Points
List of [rubric points](https://review.udacity.com/#!/rubrics/432/view) to pass the project.  

---
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
