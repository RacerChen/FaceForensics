# Step2: Split dataset into training, validation and test set

import os
import math
import shutil


def split_in_dir(dir_path, total_count, ratio):
    # ratio = [0.72, 0.14, 0.14]
    os.mkdir(dir_path + '/training')
    os.mkdir(dir_path + '/validation')
    os.mkdir(dir_path + '/test')

    training_flag = math.floor(total_count * ratio[0])
    validation_flag = training_flag + math.floor(total_count * ratio[1])
    # The left is test set
    cur_flag = 0
    for file in os.listdir(dir_path):
        if '.jpg' in file:
            print(cur_flag)
            if cur_flag < training_flag:
                shutil.move(dir_path + '/' + file, dir_path + '/training/' + file)
            elif cur_flag < validation_flag:
                shutil.move(dir_path + '/' + file, dir_path + '/validation/' + file)
            else:
                shutil.move(dir_path + '/' + file, dir_path + '/test/' + file)
            cur_flag += 1


if __name__ == '__main__':
    split_in_dir('/home/jc/Faceforensics_onServer/FaceForensics++images/Deepfakes', 1000, [0.72, 0.14, 0.14])
    split_in_dir('/home/jc/Faceforensics_onServer/FaceForensics++images/Face2Face', 1000, [0.72, 0.14, 0.14])
    split_in_dir('/home/jc/Faceforensics_onServer/FaceForensics++images/FaceSwap', 1000, [0.72, 0.14, 0.14])
    split_in_dir('/home/jc/Faceforensics_onServer/FaceForensics++images/NeuralTextures', 1000, [0.72, 0.14, 0.14])
    split_in_dir('/home/jc/Faceforensics_onServer/FaceForensics++images/x4images', 4000, [0.72, 0.14, 0.14])

