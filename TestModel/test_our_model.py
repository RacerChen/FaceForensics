# Get prediction tag from our trained model: detect all images in a directory (no face detection)
"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler

Use for test performance of this FAKE DETECTION ALGORITHM
"""
import os
import argparse
from os.path import join
import cv2
import dlib

import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from xception import xception

QT_DEBUG_PLUGINS=1


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def test_full_image_network(data_path, model_path, output_path,
                            start_frame=0, end_frame=None, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    f_prediction = open(output_path, 'a')

    # Face detector
    face_detector = dlib.get_frontal_face_detector()
    # Load model
    model = xception(num_classes=2, pretrained=False)
    # model_path = '../Model/faceforensics++_models_subset/face_detection/xception/all_c23.p'
    # choose model manually
    if model_path is not None:
        # model = torch.load(model_path, map_location={'cuda:1':'cuda:0'})
        state_dict = torch.load(model_path, map_location='cuda:0')
        model.load_state_dict(state_dict)
        # model = torch.load(model_path)
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    if cuda:
        print('CUDA!')
        model = model.cuda()

    # Frame numbers and length of output video
    frame_num = 0
    pbar = tqdm(total=end_frame-start_frame)

    for file in os.listdir(data_path):
        print(file)
        if file:
            image = cv2.imread(data_path + '/' + file)
            print(data_path + '/' + file)
            # Image size
            height, width = image.shape[:2]
            pbar.update(1)
        else:
            f_prediction.close()

        # --- Prediction ---------------------------------------------------

        # Actual prediction using our model
        prediction, output = predict_with_model(image, model,
                                                cuda=cuda)
        print('prediction:')
        print(prediction)

        f_prediction.write('Frame: ' + str(frame_num) + ', Prediction:' + str(prediction) + '\n')

        print('output:')
        print(output)

        cv2.waitKey(33)     # About 30 fps
    pbar.close()


"""
# implement param:
-i /home/jc/Faceforensics_onServer/FaceForensics++images-Big/Face2Face/test
-m /home/jc/Faceforensics_onServer/Model/xception-b5690688-after18epochs-noNT-Big.pth
-o /home/jc/Faceforensics_onServer/FaceForensics++images-Big/Face2Face/prediction_Face2Face_18epochs.txt
"""

print('CUDA is available: ' + str(torch.cuda.is_available()))
print('Device name: ' + torch.cuda.get_device_name(0))
p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--data_path', '-i', type=str)
p.add_argument('--model_path', '-mi', type=str, default=None)

p.add_argument('--output_path', '-o', type=str,
               default='.')
p.add_argument('--start_frame', type=int, default=0)
p.add_argument('--end_frame', type=int, default=1000)
p.add_argument('--cuda', action='store_true')
args = p.parse_args()

test_full_image_network(**vars(args))


