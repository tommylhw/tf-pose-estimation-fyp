Hey! This is a clone of the tf-pose-estimation by Ildoo Kim modified to work with Tensorflow 2.0+!
Link to original repo: https://github.com/jiajunhua/ildoonet-tf-pose-estimation/tree/master

# Customization for our FYP
## Setup command
1. ```conda activate AIMachine```
2. ```conda install python```
3. ```conda install tensorflow```

## Run the env
1. Browse to the root folder ```tf-pose-estimation-fyp/```.
2. The image and video src is in directory: ```images/``` and ```video/```.
3. Run the command to perform motion tracking.
   1. motion tracking for image: ```python run.py --image='[path]'```.
   2. motion tracking for video: ```python run_webcam.py --camera='[path]'```.
   3. motion tracking for realtime webcam: ```python run_webcam.py```.
4. The exported result will be stored in ```/exports/``` with created folder.