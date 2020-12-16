# This code extracts frames from video to generate dataset.

import os
import cv2


def videos2frames(video_path, frame_save_path):
    for root, dirs, files in os.walk(video_path):
        for file in files:
            cur_video = cv2.VideoCapture(os.path.join(root, file))
            success, frame = cur_video.read()  # only use the first frame
            output_file = frame_save_path + '/' + file.replace('mp4', 'jpg')
            if success:
                cv2.imwrite(output_file, frame)  # save frame
                print('Saved: ' + output_file)
            else:
                print('Error: fail to read frame!')


if __name__ == '__main__':
    videos2frames('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/videos',
                  '/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/images')
    videos2frames('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/videos/Deepfakes',
                  '/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/images/Deepfakes')
    videos2frames('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/videos/Face2Face',
                  '/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/images/Face2Face')
    videos2frames('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/videos/FaceSwap',
                  '/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/images/FaceSwap')
    videos2frames('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/videos/NeuralTextures',
                  '/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/images/NeuralTextures')
