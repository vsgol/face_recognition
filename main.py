from argparse import ArgumentParser
from time import perf_counter
from datetime import timedelta
import math

import mtcnn
import mediapipe as mp
import cv2


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("input", help='Required. An input to process. The input must be a video file')
    parser.add_argument('-o', '--output',
                        default='result',
                        help='Optional. Name of the output files to save.')
    parser.add_argument('--no_show', action='store_true',
                        help="Optional. Don't show output video.")
    parser.add_argument('--algo', default='mediapipe', choices=('haar_cascade', 'mtcnn', 'mediapipe'),
                        help='Optional. Algorithm for finding faces. Default: mediapipe.')
    return parser


def haar_cascade_handler(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return res


def mtcnn_handler(detector, frame):
    return [det['box'] for det in detector.detect_faces(frame)]


def mediapipe_handler(detector, frame):
    def normalized_coordinates(x, y, shape):
        xp = min(math.floor(x * shape[1]), shape[1] - 1)
        yp = min(math.floor(y * shape[0]), shape[0] - 1)
        return xp, yp

    detections = detector.process(frame).detections
    res = []
    for detection in detections:
        location = detection.location_data.relative_bounding_box
        res.append(normalized_coordinates(location.xmin, location.ymin, frame.shape) +
                   normalized_coordinates(location.width, location.height, frame.shape))
    return res


def main():
    args = build_argparser().parse_args()
    video_path = args.input
    algo = {
        'mtcnn': (mtcnn.MTCNN(), mtcnn_handler),
        'haar_cascade': (cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml'), haar_cascade_handler),
        'mediapipe': (mp.solutions.face_detection.FaceDetection(), mediapipe_handler)
    }
    detector, detector_handler = algo[args.algo]

    start_time = perf_counter()

    in_video = cv2.VideoCapture(video_path)
    fps = in_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(in_video.get(3))
    frame_height = int(in_video.get(4))
    out_video = cv2.VideoWriter(args.output + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))
    out_file = open(args.output + '.txt', 'w')

    second = 0
    frame_num = 0
    while True:
        ret, frame = in_video.read()
        if not ret:
            break

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector_handler(detector, frame)
        if frame_num >= second * fps:
            if len(faces) > 0:
                out_file.write(f'{timedelta(seconds=second)} faces: '
                               + ' '.join([f'(({x}, {y}), ({x + w}, {y + h}))' for (x, y, w, h) in faces])
                               + '\n')
            print(f'{timedelta(seconds=second)} of video processed')
            second += 1
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame_num += 1

        # Write the resulting frame
        out_video.write(frame)
        if not args.no_show:
            cv2.imshow('Face recognition', frame)
            key = cv2.waitKey(1)
            # Quit
            if key in {ord('q'), ord('Q'), 27}:
                break

    end_time = perf_counter()
    print(f'Completed in {end_time - start_time:.2f} seconds')
    in_video.release()
    out_video.release()
    out_file.close()
    if hasattr(detector, 'close'):
        detector.close()


if __name__ == '__main__':
    main()
