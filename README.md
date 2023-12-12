## Introduction
The goal of this project was to train an instance detection and segmentation model on a custom dataset to segment instances of butterflies from their backgroundz and surrounding environments. This model was built by training a YOLOv8 segmentation model from ultralytics on a custom dataset.
## Data Selection
The dataset I chose to use for training and testing this model is a modified version of the "Nature" dataset by [Ayoola Olafenwa](https://github.com/ayoolaolafenwa), found [here](https://github.com/ayoolaolafenwa/PixelLib/releases/tag/1.0.0) as a part of the [PixelLib](https://github.com/ayoolaolafenwa/PixelLib) project. This dataset contains a total of 400 high quality images of butterflies annotated for instance segmentation, split into 300 images for training and 100 for testing. These images are originally labeled using the LabelMe JSON format, but YOLO uses a different annotation format, so a tool called [labelme2yolo](https://pypi.org/project/labelme2yolo/) was used to convert this dataset into a format and file structure that YOLO can be trained on.
## Tools Used
* [labelme2yolo python package](https://pypi.org/project/labelme2yolo/) for converting labelme formatted/annotated "Nature" dataset to YOLO formatted dataset with accompanying dataset.yaml
* [ultralytics yolo package (command line and python)](https://github.com/ultralytics/ultralytics) for training a custom model and inferencing on test images/video.
* [python venv library](https://docs.python.org/3/library/venv.html) for virtual environment setup.
## Methods
### Dataset Modification & Training
1. The "Nature" dataset was downloaded extraneous instances were removed, as it originally contained images of both squirrels and butterflies, and I wanted to make a somewhat simpler model focused on a single class for both detection and segmentation using YOLO.
2. Once all images and annotations for squirrels were removed from both the test and training sets, labelme2yolo was used to convert both train and test sets to YOLO format with the `--val_size 0` and `--test_size 0` arguments to avoid the tool furter splitting the data, as the dataset was already split up into train and test sets.
3. The `dataset.yaml` file created by these commands was then modified to point to the resulting train and test sets and put in the root project directory.
4. Once the dataset reconfiguration and restructuring were complete, the model was trained by running [train.bat](https://github.com/shepherdm1atwit/butterfly-segmentation/blob/main/train.bat), which trains a model on top of [yolov8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) (the medium size of yolov8's segmentation models) with a maximum of 200 epochs and a batch size of 4. All other arguments and training parameters were left as default and can be found in [args.yaml](https://github.com/shepherdm1atwit/butterfly-segmentation/blob/main/runs/segment/train/args.yaml).
5. The resulting model was evaluated using it's mAP50-95 value, made up of an average of mean Average Precision values over multiple Intersection over Union threshholds from 0.5 to 0.95.
## Analysis/Results
The model resulting from this training process (saved as [yolov8_butterfly_custom.pt](https://github.com/shepherdm1atwit/butterfly-segmentation/blob/main/yolov8_butterfly_custom.pt)) is quite accurate, achieving a mAP50-95 of 91.41% and a validation segmentation loss of 0.7888 (down from a starting loss of 3.0348) and a bounding box (detection) loss of 0.42515 (down from 1.2523). All loss, accuracy, and related values for each epoch can be found [here](https://github.com/shepherdm1atwit/butterfly-segmentation/blob/main/runs/segment/train/results.csv), and graphs of these values, as well as several example images segmented with this model can be found below.  
![results](https://github.com/shepherdm1atwit/butterfly-segmentation/blob/main/runs/segment/train/results.png)  
![example1](https://github.com/shepherdm1atwit/butterfly-segmentation/blob/main/runs/segment/predict/butterfly%20(2).png)![example2](https://github.com/shepherdm1atwit/butterfly-segmentation/blob/main/runs/segment/predict/butterfly%20(40).png)![example3](https://github.com/shepherdm1atwit/butterfly-segmentation/blob/main/runs/segment/predict/butterfly%20(102).png)

Using the YOLO package from ultralytics, videos can also be segmented simply be specifying the video instead of an image as the source in `model.predict()`. An example of this can be found [here](https://github.com/shepherdm1atwit/butterfly-segmentation/blob/main/runs/segment/predict2/butterfly.avi).
### Deployment/Inferencing
To use this model locally, follow these steps:
1. Clone this repository and enter it's directory:
   ```sh
   git clone https://github.com/shepherdm1atwit/butterfly-segmentation
   cd butterfly-segmentation
   ```
2. Create a virtual environment using venv and install requirements:
   ```sh
   python -m venv ./venv
   pip install -r requirements.txt
   ```
   If you would like to use GPU for inference, make sure to install pytorch with cuda support (instructions [here](https://pytorch.org/get-started/locally/) example for cuda 12.1 on windows below):
   ```sh
   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. Inference on 10 random images from the test set:
   ```sh
   python ./predict_10_random.py
   ```
   or on [butterfly.mp4](https://github.com/shepherdm1atwit/butterfly-segmentation/blob/main/butterfly.mp4):
   ```sh
   python ./predict_video.py
   ```
4. To inference on other images or videos, the following Python can be used:
   ```python
   from ultralytics import YOLO

   model = YOLO("yolov8_butterfly_custom.pt")
   model.predict(source="[image_or_video_path_here]", show=True, save=True)
   ```
