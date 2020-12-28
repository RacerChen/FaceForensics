# Step3: Merge dataset of manipulated images and original images

import os
import shutil


def merge_in_one(split_dataset_path, merge_dataset_path):
    for level1_dir in os.listdir(split_dataset_path):
        # ['FaceSwap', 'Deepfakes', 'x4images', 'NeuralTextures', 'Face2Face']
        if level1_dir == 'x4imagesV2':
            tag = '0'  # real face
        else:
            tag = '1'  # fake face
        print('In dir: ' + level1_dir)
        cur_full_path = split_dataset_path + '/' + level1_dir
        for level2_dir in os.listdir(cur_full_path):
            # ['test', 'train', 'val']
            cur_full_path = split_dataset_path + '/' + level1_dir + '/' + level2_dir
            # print(level2_dir)
            for file in os.listdir(cur_full_path):
                # file ful path
                new_filename = level1_dir + '_' + file
                print(new_filename)
                if level2_dir == 'test':
                    shutil.copy(cur_full_path + '/' + file, merge_dataset_path + '/test/' + tag + '/' + new_filename)
                if level2_dir == 'train':
                    shutil.copy(cur_full_path + '/' + file, merge_dataset_path + '/train/' + tag + '/' + new_filename)
                if level2_dir == 'val':
                    shutil.copy(cur_full_path + '/' + file, merge_dataset_path + '/val/' + tag + '/' + new_filename)


if __name__ == '__main__':
    merge_in_one('/home/jc/Faceforensics_onServer/FaceForensics++images-Big',
                 '/home/jc/Faceforensics_onServer/Final_Faceforensics++no_NT-Big')
