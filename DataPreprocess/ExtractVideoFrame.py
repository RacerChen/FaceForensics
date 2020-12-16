# This code extracts frames from video to generate dataset.

import os
import cv2


def videos2frames(video_path, frame_save_path, frame_to_read):
    for root, dirs, files in os.walk(video_path):
        for file in files:
            cur_video = cv2.VideoCapture(os.path.join(root, file))
            frame_num = 0
            while frame_num < frame_to_read:
                success, frame = cur_video.read()  # only use the first frame
                if frame_to_read == 1:
                    output_file = frame_save_path + '/' + file.replace('mp4', 'jpg')
                else:
                    output_file = frame_save_path + '/' + str(frame_num) + '_' + file.replace('mp4', 'jpg')
                    gap = 0
                    while gap < 24:
                        success, frame = cur_video.read()  # avoid similarity between continuous frames
                        gap += 1
                if success:
                    cv2.imwrite(output_file, frame)  # save frame
                    print('Saved: ' + output_file)
                else:
                    print('Error: fail to read frame!')
                frame_num += 1


if __name__ == '__main__':
    videos2frames('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/original_sequence/videos',
                  '/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/original_sequence/x4images', 4)
    # videos2frames('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/videos/Deepfakes',
    #               '/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/images/Deepfakes', 1)
    # videos2frames('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/videos/Face2Face',
    #               '/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/images/Face2Face', 1)
    # videos2frames('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/videos/FaceSwap',
    #               '/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/images/FaceSwap', 1)
    # videos2frames('/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/videos/NeuralTextures',
    #               '/home/jc/Faceforensics_onServer/FaceForensicsDATASET/c23/manipulated_sequences/images/NeuralTextures', 1)
