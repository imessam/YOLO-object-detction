# YOLO object detection model PyTorch implementation from scratch.
- Using the MobileNet pretrained model as a backend to extract features from the input images.
- Trained on the VOC 2007 dataset train and validation splits plus VOC 2012 dataset train split and the validation split for validation.
- Tested on the VOC 2007 dataset test split.
- AP@IOU(0.4) = 41.67% , AP@IOU(0.5) = 40.95% , AP@IOU(0.6) = 34.16%
- With Non Max Supression IOU threshold : 0.7, AP@IOU(0.4) = 45.8% , AP@IOU(0.5) = 44.57% , AP@IOU(0.6) = 36.74%
- Highest AP@IOU(0.1) with Non Max Supression IOU threshold 0.7 = 46.49%
- To run the model, python.exe run.py "image-path"