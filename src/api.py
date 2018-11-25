import argparse
import base64
import cv2 as cv
import numpy as np
from flask import Flask, request, jsonify, Response
import common
from pose import get_pose_data, compare
from modelsettings import coco, mpi


PARSER = argparse.ArgumentParser()

PARSER.add_argument('--proto', help='Path to the .prototxt file')
PARSER.add_argument('--model', help='Path to the .caffemodel file')
PARSER.add_argument('--dataset', choices=['COCO', 'MPI'], help='Specify what kind of model was trained. It could be (COCO, MPI) depends on the dataset.')
PARSER.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')

ARGS = PARSER.parse_args()

if ARGS.dataset == 'COCO':
    DATASET_INFO = (coco.BODY_PARTS, coco.POSE_PAIRS, coco.PAIR_COLORS)
else:
    DATASET_INFO = (mpi.BODY_PARTS, mpi.POSE_PAIRS, mpi.PAIR_COLORS)


NETWORK = cv.dnn.readNetFromCaffe(common.find_file(ARGS.proto), common.find_file(ARGS.model))

APP = Flask(__name__)


@APP.route('/api/compare', methods=['POST'])
def run_comparison():
    data = request.get_json()

    if data is None:
        return jsonify({'error': 'No valid request body, json missing!'})
    else:

        frame = base64_to_image(data['frame'])
        template = base64_to_image(data['template'])

        _, frame_vectors = get_pose_data(frame, ARGS.thr, NETWORK, DATASET_INFO)
        _, template_vectors = get_pose_data(template, ARGS.thr, NETWORK, DATASET_INFO)

        return jsonify(compare(frame_vectors, template_vectors))


@APP.route('/api/template', methods=['POST'])
def create_template():
    data = request.get_json()

    if data is None:
        return jsonify({'error': 'No valid request body, json missing!'})
    else:
        template = base64_to_image(data['template'])

        template_points, _ = get_pose_data(template, ARGS.thr, NETWORK, DATASET_INFO)

        body_parts, pose_pairs, _ = DATASET_INFO

        white_background = np.full(template.shape, 255, np.uint8)
        pose_pairs_color = [(0, 0, 0)] * len(pose_pairs)

        common.draw_vectors(template_points, pose_pairs, body_parts, pose_pairs_color, white_background)

        _, template_image = cv.imencode('.jpg', white_background)

        return Response(response=template_image.tostring(), status=200, mimetype="image/jpeg")


def base64_to_image(encoded_image):
    encoded_data = encoded_image.split(',')[1]
    encoded_data_as_byte_array = bytes(encoded_data, 'utf-8')
    image_array = np.frombuffer(base64.decodebytes(encoded_data_as_byte_array), np.uint8)

    return cv.imdecode(image_array, cv.IMREAD_COLOR)


# start flask app
APP.run(host="0.0.0.0", port=5000)
