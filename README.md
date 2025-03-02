# Getting Started 
## Installation
you can create a new Conda environment using:

'''base
conda env create -f FNB.yaml
'''

## Dataset
1. Please download and fill out the application form ('Medical Dataset Access Application Form') and send it to the author's email address(yx.chen@whu.edu.cn). 
2. Place the obtained dataset into the 'data' folder.

## Training
Modify the corresponding dataset path in the file (tools/config.py).
'''base
1. python ./detection_method/train_detection.py # train detection model
2. python train.py  # train segmentation model
3. python ./class_method/train_class.py # train classification model
'''

## Testing
FNB-ADS inference code can run the file
'''base
python main.py
'''

Thank you for your interest in FNB-ADS. We have packaged FNB-ADS as a web page that allows you to upload files, perform inference, and download the results.

## Infer Demonstrations

### Ultrasound picture FNB reasoning effect demonstration

[https://raw.githubusercontent.com/SIGMACX/FNB-AD/FNB-ADS/infer_results_images/image_infer_results.mp4](https://github.com/user-attachments/assets/974d0eb9-e434-4ff7-8d90-8212acc29037)


### Ultrasound video FNB reasoning effect demonstration

[https://raw.githubusercontent.com/SIGMACX/FNB-AD/FNB-ADS/infer_results_images/infer_video_results.mp4](https://github.com/user-attachments/assets/e8d0efc4-e804-46c5-8e40-44441c41e1d5)

 
