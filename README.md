
### Sefl-Driving Car Nanodegree Program. Term 1
<img style="float: left;" src="https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg">

## Project 5: Vehicle Detection and Tracking

### Overview
---
The goal of this project is to create a software pipeline to identify vehicles in a video from a front-facing camera on a car.

The project includes following procedures:
* Perform a [Histogram of Oriented Gradients (HOG)](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector (optionally).
* Normalize extracted features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the [test_video.mp4](test_video.mp4) and later implement on full [project_video.mp4](project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

A detail witeup of the projects can be found in [writeup_report.md](writeup_report.md) document.

### Project directory content:

* [README.md](README.md) - This file.
* [vehicle_detection.ipynb](vehicle_detection.ipynb) -  IPython Jupyter notebook with the project code.
* [p5lib/](p5lib/) folder - Python modules imported by the notebook:
   - [features.py](p5lib/features.py) - a module with feature extraction functions
   - [detection.py](p5lib/detection.py) -a module with vehicle detection functions
   - [data_preparation.py](p5lib/data_preparation.py) - data load/explore helper function
   - [visualize.py](p5lib/visualize.py) - visualization helper functions
* [writeup_report.md](writeup_report.md) - The project writeup - a markdown file that includes the [rubric](https://review.udacity.com/#!/rubrics/513/view) points as well as description of how each point was addressed in the project.
* [project_video_output.mp4](project_video_output.mp4) - The final video output.
* [output_images/](output_images/) - A folder with examples of the output from each stage of the processing pipeline 

### Project Environment

The project environment was created with [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit).
