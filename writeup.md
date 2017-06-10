##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* The color features: will use the histograms of color to produce a classifier for car detection
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* The heatmap and labelizaion: those which is used for handling the Multiple Detections and False Positives
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/color_feature.png
[image2]: ./examples/spin_feature.png
[image3]: ./examples/car_not_car.png
[image4]: ./examples/hog_feature.png
[image5]: ./examples/hog_visualization.png
[image6]: ./examples/sliding_out.png
[image7]: ./examples/detected_image.png
[image8]: ./examples/heat_map.png
[image9]: ./examples/label_image.png
[video1]: ./project_video.mp4

---
###Color Feature and Binned Features

#### Explain how you extracted Color Feature and Binned Features from the training images.

The different object has different size of colors or different value of colors. Sometimes we can identify the object through the their special colors. So I define the function in the third, forth, fifth code cell of the IPython notebook in `ProjectSolution.ipynb` file. The color feature of image is like below:

![alt text][image1]

The different size of image show different infomation, but we also identify the object from different size image. Like below:

![alt text][image2]

So I define the function to extract these features.
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code in the second code cell of the IPython notebook is got the all data image for training. I get the data from [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image3]

I defined the function `get_hog_features` to extract the hog feature of image. The feature is like below:

![alt text][image4]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image5]

####2. Explain how you settled on your final choice of HOG parameters.

To get the best effect for hog feature, I use SVM method to train and test the model. I choose the one which has most high accuracy for test data. Details will be introduced below. I define the function in the third code cell of the IPython notebook in `pipeline.ipynb` file. The examples used is like below:

|Index|Color Space|Orientation|pexels_per_cell|cells_per_block|hog_channel|accuracy| 
|:---------------------:|:---------------------------------------------:| 
|1|RGB|9|8|(2,2)|ALL|0.9631|
|2|HSV|9|8|(2,2)|ALL|0.9783|
|3|LUV|9|8|(2,2)|ALL|0.9725|
|4|HLS|9|8|(2,2)|ALL|0.9811|
|5|YUV|9|8|(2,2)|ALL|0.9801|
|6|YCrCb|9|8|(2,2)|ALL|0.9875|
|7|YCrCb|8|8|(2,2)|ALL|0.9865|
|8|YCrCb|7|8|(2,2)|ALL|0.9783|
|9|YCrCb|10|8|(2,2)|ALL|0.9792|
|10|YCrCb|9|7|(2,2)|ALL|0.9803|
|11|YCrCb|9|9|(2,2)|ALL|0.9721|
|12|YCrCb|9|8|(3,3)|ALL|0.978|
|13|YCrCb|9|8|(4,4)|ALL|0.9738|
|14|YCrCb|9|8|(2,2)|0|0.9454|
|15|YCrCb|9|8|(2,2)|1|0.9265|
|16|YCrCb|9|8|(2,2)|2|0.9105|

From this table, I finally choose the best params like color space using YCrCb, etc.

####3. Describe how you trained a classifier using your selected HOG features and color features.

I trained a linear SVM using HOG features, color features and binned features. In function `combine_img_features`, I solve this three features combination. And using the best example params above to train the model. I use the linear SVC of SVM to train my model and decide the param C to be 1.0. I define the function in the Step fifth in code cells of the IPython notebook in `ProjectSolution.ipynb` file.

###Sliding Window Search

####1. Describe how you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at given scales in special position of the image. In the Step forth in code cells of the IPython notebook in `ProjectSolution.ipynb`, I defined my work functions. The image which is used to handle the slide windons like below:

![alt text][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I defined the function `search_windows` which use svc model to predict whether car or not. And I defined the function `search_multi_scales` which use four scales to predict whether car or not in given image. The scales is 70px, 120px, 180px and 240px. Here is a detected example image:

![alt text][image7]

Because a true positive is typically accompanied by several positive detections, while false positives are typically accompanied by only one or two detections, a combined heatmap and threshold is used to differentiate the two. The add_heat function which I define increments the pixel value (referred to as "heat") of an all-black image the size of the original image at the location of each detection rectangle. Areas encompassed by more overlapping rectangles are assigned higher levels of heat. The following image is the resulting heatmap from the detections in the image above:

![alt text][image8]

Use different threshold can solve the multiple detectios and false positives.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For multiple detection and false position, I used the heat map to decided whther cut area is used or not and scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a frame of video, the result of scipy.ndimage.measurements.label() and the bounding boxes then overlaid on the last frame of video:

![alt text][image9]

For pipeline in video, I define a class `TrackBoxes` to solve the frame of video. This class will save multiple boxes (we can change) of detected in a frame of image. And keep the boxes for multi frames to use in order to improve the correct position of car.


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems that I faced while implementing this project were mainly concerned with detection accuracy. Different params aim to different accuracy in multi features. I try the all params in different combination. Balancing the accuracy of the classifier with execution speed was crucial. Scanning 100 test images using a classifier that achieves 98% accuracy should result in around 2 misidentified windons per frame of video. But for multi frames of close time, we can treat them as same position for car (if there is). So we can use the boxes if current frame to apply for next frame. But we should keep in mind that the number of boxes which should use.

The other problem is that the image colorspace and scaling due to the difference in how cv2 and matplotlib load images. For `png` image and `jpg` image, there are different scales in data. Refer to [this question](https://discussions.udacity.com/t/svm-does-not-work-for-cut-image-from-test-image/262665). I was stuck for this because of my lack in these knowlege.

Of course, we can use other ML method to train the model like NN. In the program, we use NN to train the model for traffic-fign detection. Maybe this will work fine for this, I think. I will have a try for this.
