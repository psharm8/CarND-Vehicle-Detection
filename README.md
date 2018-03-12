## Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labelled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG.png
[image2]: ./output_images/simple_bbox.png
[image3]: ./output_images/heatmaps.png
[image4]: ./output_images/heatmap_threshold_bbox.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cells 2 through 9 of the IPython notebook `vehicle_detection.jpynb`. Cell 9 makes the actual call to extract features on all training images.

I started by reading in all the `vehicle` and `non-vehicle` images and then passing them through the `extract_features(...)` method defined in code cell 7 which utilizes the function `get_hog_features(...)` from code cell 3.

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like (code cell 4).

Here is an example using the Red channel from `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

After some tweaking of the parameters I went back to the values introduced during lesson videos as they worked well without making the feature vector too long. I decided to include spatial and histogram features with the HOG features as it increased the model accuracy. I noticed that YCrCb color space along with spatial and histogram features gave a good test accuracy of about 98.7%.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In code cell 10 I first shuffled and made a 90-10 split of data into training and test set. Since I was using three feature sets together I used the `StandardScaler()` to normalize the feature vectors.


Then I trained a linear SVM using `LinearSVC()` from sklearn library (code cell 11) and dumped the trained model to a pickle file `model_info.p`. The pickle also included various parameters that were required for the extraction of features. This helped me train the model once to my satisfaction and then repeat the detection part only by changing the detection related parameters such as window overlaps, scales, heatmap thresholds, etc.

Here is a sample of simple car detection without any measures to suppress the false positives or multiple detection.

![alt text][image2]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

As introduced in the lesson material, I used the sliding window stepping 2 cells for every window slide. I limited the window ROI to the lower half of the frame as we won't expect cars in the upper half (at least for this project video).

For the scales, it took a while, but I figured that training samples were 64x64 but the cars on 1280x720 frame would mostly appear larger than 64x64. So, I set multiple scales from 1.1 to 2.5 (4 scales). It increased the process time for each frame considerably, but the detection was good. It takes about 2.3 seconds per frame to detect, heatmap, threshold and draw boxes. 


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately, I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then added it to a list of recent heatmaps, I used this list to average the heatmaps over last 5 frames and then thresholded that average to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six heatmaps of test images:

![alt text][image3]


### Here the resulting bounding boxes are drawn after labelling:
![alt text][image4]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first thing that comes to mind about this implementation is the time it takes to process each frame. It is nowhere near real-time and cannot be used for vehicle detection for self-driving cars in its current form. There are potential places where multithreading or even GPU compute (like CUDA or OpenCL) can be utilized to extract the feature vectors efficiently. 

Furthermore, at about 41 seconds in, the video has a spot on the left with a tree shadow which I could not manage to exclude as a false positive without affecting the true detections. I believe that can be fixed with more robust training data or using deep-learning concepts as in traffic sign classifier which outperformed car detection using Linear SVM.


Another potential pitfall would be a steep up-hill drive where the cars may appear on the upper half of the frame. This means, increasing the search space and in-turn increasing the detection time.  

