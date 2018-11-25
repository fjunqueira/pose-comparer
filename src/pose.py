import cv2 as cv
import numpy as np


def compare(frame_vectors, template_vectors):
    return [dot_or_none(i, t) for i, t in zip(frame_vectors, template_vectors)]


def dot_or_none(vec1, vec2):
    return np.dot(vec1, vec2) if vec1 is not None and vec2 is not None else None


def get_pose_data(image, thr, network, dataset_info):

    body_parts, pose_pairs, _ = dataset_info

    heat_maps = compute_heat_maps(network, image)
    pose_points = get_pose_points(body_parts, heat_maps, image.shape, thr)
    pose_vectors = get_pose_vectors(pose_points, pose_pairs, body_parts)

    return (pose_points, pose_vectors)


def compute_heat_maps(neural_network, frame, input_dimensions=(368, 368)):

    input_width, input_height = input_dimensions

    input_blob = cv.dnn.blobFromImage(frame, 1.0 / 255, (input_width, input_height), (0, 0, 0), swapRB=False, crop=False)
    neural_network.setInput(input_blob)
    return neural_network.forward()


def get_pose_points(body_parts, heat_maps, frame_dimensions, threshold):
    points = []

    for i in range(len(body_parts)):

        # Slice heat_map of corresponging body's part.
        heat_map = heat_maps[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        frame_height, frame_width, _ = frame_dimensions

        _, confidence, _, point = cv.minMaxLoc(heat_map)

        x_coord = (frame_width * point[0]) / heat_maps.shape[3]
        y_coord = (frame_height * point[1]) / heat_maps.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append([int(x_coord), int(y_coord)] if confidence > threshold else None)

    return points


def get_pose_vectors(points, pose_pairs, body_parts):
    normalized_vectors = []

    for pair in pose_pairs:

        part_from = pair[0]
        part_to = pair[1]

        id_from = body_parts[part_from]
        id_to = body_parts[part_to]

        if points[id_from] and points[id_to]:

            vector = np.array(points[id_to]) - np.array(points[id_from])
            normalized_vectors.append(vector / np.linalg.norm(vector, axis=0))
        else:
            normalized_vectors.append(None)

    return normalized_vectors
