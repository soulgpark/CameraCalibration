import cv2
import numpy as np
import os

data = np.load('camera/data/calibration_result.npz')
mtx = data['mtx']
dist = data['dist']

print("Loaded camera matrix:\n", mtx)
print("Loaded distortion coefficients:\n", dist)

cap = cv2.VideoCapture('camera/data/chessboard.avi')

output_path = 'camera/data/undistorted_output.mp4'
if not os.path.exists('camera/data'):
    os.makedirs('camera/data')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    cv2.imshow('Undistorted', undistorted_frame)

    if out is None:
        height, width = undistorted_frame.shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    out.write(undistorted_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
