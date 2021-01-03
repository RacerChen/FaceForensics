import os
import cv2
import dlib

from detect_from_video import get_boundingbox


def extract_face(frame_dir, output_dir):
    for file in os.listdir(frame_dir):
        cur_filename = frame_dir + '/' + file
        image = cv2.imread(cur_filename)
        # print(cur_filename)
        height, width = image.shape[:2]

        # Init face detector
        face_detector = dlib.get_frontal_face_detector()

        # Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)

        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]
            cv2.imwrite(output_dir + '/' + file, cropped_face)
            print(output_dir + '/' + file)


tag = 1
for set_type in ['test', 'train', 'val']:
    for face_fake_type in ['0', '1']:
        cur_subdir = set_type + '/' + face_fake_type
        frame_dir = '/home/jc/Faceforensics_onServer/Final_Faceforensics++no_NT-Big/' + cur_subdir
        output_dir = '/home/jc/Faceforensics_onServer/Final_Faceforensics++no_NT-Big_facecorp/' + cur_subdir
        print(tag)
        tag += 1
        extract_face(frame_dir, output_dir)
# extract_face('/home/jc/Faceforensics_onServer/Final_Faceforensics++no_NT-Big/test/0',
#              '/home/jc/Faceforensics_onServer/Final_Faceforensics++no_NT_facecorp/test/0')








