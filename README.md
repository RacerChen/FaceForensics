# FaceForensics++ DIY Version

## Introduction

This project fork from https://github.com/ondyari/FaceForensics.

It is convenient to implement their trained model by this original project.

This forked DIY version provide TOOLS for:

### Data Preprocessing

In directory [/FaceForensics/DataPreprocess]:

* [ExtractVideoFrame.py] : Extract frame from provided video dataset.

* [ExtractFaceFromFrame.py] : Extract face from every frame. Because the paper introduce a face detection algorithm to enhance the performance of fake detection.

* [SplitDataset.py] : Split dataset into training, validation and test set. The ratio mentioned in their paper is 0.72. 0.14, 0.14.

* [MergeDataset.py] : Merge dataset of manipulated images(Deepfake, Face2face, Faceswap and NeuralTextures) into type 1 and original images into type 0. 

### Model Training

In directory [/FaceForensics/TrainModel]:

* [TrainModel_2.py] : Use a rransfer learning template in pytorch.
  * Choose your merged dataset path in Line 25
  * Choose batch size in line  37
  * Pretrain the last linear layer with 3 epochs by adding parameter '----pretrain3epochs' like `TrainModel_2.py --pretrain3epochs`
  * Train whole model with 15epochs directly `TrainModel_2.py`
  *  Save the trained model in Line 199(specify your output path and filename here).

### Model Testing

You can see the testing and validation accuracy in the console during training time. If you want to see the test set counterpart, use code in directory [/FaceForensics/TestModel]:

* [test_model.py] : Detect all images in a directory and write all prediction output into a .txt file.
* [test_model_GUI.py] : Detect single video and output a series of frames with face box and prediction tag.
* [test_our_model.py] : Implement our trained model of [TrainModel_2.py] and output prediction tag into .txt file. Implement format is : `test_our_model.py -i /test_set_images_path -m /model_path -o /output_txt_file_path`

### Stat Accuracy

After get prediction output by [test_our_model.py], use code in directory  [/FaceForensics/StatResult] to stat accuracy:

* [cal_accuracy.py] : stat whole test set accuracy.
* [cal_accuracy_single.py] : stat single test set accuracy, eg. Deepfakes. 

## Last but not least

If you have any questions, contact me via george_chenjiajie@qq.com.

