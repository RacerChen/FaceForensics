# Step2: Split dataset into training, validation and test set

import os
import math
import shutil


def split_in_dir(dir_path, total_count, ratio):
    # ratio = [0.72, 0.14, 0.14]
    os.mkdir(dir_path + '/train')
    os.mkdir(dir_path + '/val')
    os.mkdir(dir_path + '/test')

    training_flag = math.floor(total_count * ratio[0])
    validation_flag = training_flag + math.floor(total_count * ratio[1])
    # The left is test set
    cur_flag = 0
    for file in os.listdir(dir_path):
        if '.jpg' in file:
            print(cur_flag)
            if cur_flag < training_flag:
                shutil.move(dir_path + '/' + file, dir_path + '/train/' + file)
            elif cur_flag < validation_flag:
                shutil.move(dir_path + '/' + file, dir_path + '/val/' + file)
            else:
                shutil.move(dir_path + '/' + file, dir_path + '/test/' + file)
            cur_flag += 1


if __name__ == '__main__':
    split_in_dir('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/imagesV2/Deepfakes',
                 32000, [0.72, 0.14, 0.14])
    split_in_dir('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/imagesV2/Face2Face',
                 32000, [0.72, 0.14, 0.14])
    split_in_dir('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/imagesV2/FaceSwap',
                 32000, [0.72, 0.14, 0.14])
    # split_in_dir('/home/jc/Faceforensics_onServer/FaceForensics++images/NeuralTextures', 1000, [0.72, 0.14, 0.14])
    split_in_dir('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/original_sequence/x4imagesV2',
                 96000, [0.72, 0.14, 0.14])

