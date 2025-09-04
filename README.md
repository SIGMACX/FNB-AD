# Getting Started 
## Installation
you can create a new Conda environment using:

```
conda env create -f FNB.yaml
```

## Training
Modify the corresponding dataset path in the file (tools/config.py).

```
1. python ./detection_method/train_detection.py # train detection model
2. python train.py  # train segmentation model
3. python ./class_method/train_class.py # train classification model
```

## Testing
FNB-ADS inference code can run the file

```
python main.py
```

## Web Inference Setup

1. Download the model weights from [Google Drive](https://drive.google.com/file/d/18QKP3dPUVskKHPpwEvVKYYdFVLYol65y/view?usp=drive_link).  
2. Download `web.zip` as well.  
3. Unzip `models.zip` and place the extracted folder inside the `web` directory, making sure it is at the same level as `app.py`.  
4. Make sure to update any file paths in the code if necessary (e.g., paths pointing to the models or web assets).  
5. Run the `app.py` file to start testing.



Thank you for your interest in FNB-ADS. We have packaged FNB-ADS as a web page that allows you to upload files, perform inference, and download the results.

## Infer Demonstrations

### Ultrasound picture FNB reasoning effect demonstration

[https://raw.githubusercontent.com/SIGMACX/FNB-AD/FNB-ADS/infer_results_images/image_infer_results.mp4](https://github.com/user-attachments/assets/974d0eb9-e434-4ff7-8d90-8212acc29037)


### Ultrasound video FNB reasoning effect demonstration

[https://raw.githubusercontent.com/SIGMACX/FNB-AD/FNB-ADS/infer_results_images/infer_video_results.mp4](https://github.com/user-attachments/assets/e8d0efc4-e804-46c5-8e40-44441c41e1d5)

 
