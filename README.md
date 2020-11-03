# mbfacenet_tf2 with Coral EdgeTPU Support 

Mobilefacenet with Tensorflow-2, EdgeTPU models also supplied for running model on Coral EdgeTPU

## Introduction
Tensorflow 2 version of mobilefacenet from [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)

## Demo
[Video Link](https://www.youtube.com/watch?v=o6G-xXyHyAM)

## Usage

### Part 1: Preprocessing

* All images of dataset are preprocessed following the [SphereFace](https://github.com/wy1iu/sphereface) and you can download the aligned images at [Align-CASIA-WebFace@BaiduDrive](https://pan.baidu.com/s/1k3Cel2wSHQxHO9NkNi3rkg) and [Align-LFW@BaiduDrive](https://pan.baidu.com/s/1r6BQxzlFza8FM8Z8C_OCBg).

### Part 2: Train

  1. Change the **CAISIA_DATA_DIR** and **LFW_DATA_DAR** in `config.py` to your data path.
  
  2. Train the mobilefacenet model. 
  
        **Note:** The default settings set the batch size of 512, use 2 gpus and train the model on 70 epochs. You can change the settings in `config.py`
      ```
      python3 trainface.py
      ```
      
### Part 3: Test

  1. Test the model on LFW.
    
      **Note:** I have tested `lfw_eval.py` on the caffe model at [SphereFace](https://github.com/wy1iu/sphereface), it gets the same result.
    
      ```
      python3 lfw_eval.py --resume --feature_save_dir
      ```
      * `--resume:` path of saved model
      * `--feature_save_dir:` path to save the extracted features (must be .mat file)

### Results

  * You can just run the `lfw_eval.py` to get the result, the accuracy on LFW like this:

  | Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | AVE(ours) | Paper(112x96) |
  | ------ |------|------|------|------|------|------|------|------|------|------| ------ | ------ |
  | ACC | 99.00 | 99.00 | 99.00 | 98.67 | 99.33 | 99.67 | 99.17 | 99.50 | 100.00 | 99.67| **99.30** | 99.18 |


### Dataset
Use the same dataset as used in [Mobilefacenet-Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) to train. CASIA is used for training and LFW is used for testing.

### Training
Change the directory pointing to image dataset in train.py. I trained the model directly with ArcFace by setting RESUME to False but it is worthwhile to try out pretraining with softmax loss

### Result
Trained model is evaluate on each epoch use LFW dataset and I got 99.3% accuracy without pretraining

# Credit to
1. Playground by Qihang Zheng: https://github.com/zhen8838/playground
2. Mobilefacenet-Pytorch: https://github.com/Xiaoccer/MobileFaceNet_Pytorch
3. MobileFaceNet-Keras: https://github.com/TMaysGGS/MobileFaceNet-Keras
