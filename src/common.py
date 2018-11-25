import os
import cv2 as cv


def find_file(filename):
    if filename:
        if os.path.exists(filename):
            return filename

        fpath = cv.samples.find_file(filename, False)
        if fpath:
            return fpath

        samples_data_dir = os.path.join(os.path.dirname(os.path.abs_path(__file__)),
                                        '..',
                                        'data',
                                        'dnn')
        if os.path.exists(os.path.join(samples_data_dir, filename)):
            return os.path.join(samples_data_dir, filename)

        for path in ['OPENCV_DNN_TEST_DATA_PATH', 'OPENCV_TEST_DATA_PATH']:
            try:
                extra_path = os.environ[path]
                abs_path = os.path.join(extra_path, 'dnn', filename)
                if os.path.exists(abs_path):
                    return abs_path
            except KeyError:
                pass

        print('File ' + filename + ' not found! Please specify a path to '
              '/opencv_extra/testdata in OPENCV_DNN_TEST_DATA_PATH environment '
              'variable or pass a full path to model.')
        exit(0)


def draw_vectors(points, pose_pairs, body_parts, pair_colors, frame):
    for pair, color in zip(pose_pairs, pair_colors):
        part_from = pair[0]
        part_to = pair[1]

        id_from = body_parts[part_from]
        id_to = body_parts[part_to]

        if points[id_from] and points[id_to]:
            cv.line(frame, tuple(points[id_to]), tuple(points[id_from]), color, 3)
            cv.ellipse(frame, tuple(points[id_to]), (5, 5), 0, 0, 360, color, cv.FILLED)
