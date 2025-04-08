import cv2
import numpy as np
import os

chessboard_size = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

cap = cv2.VideoCapture('camera/data/chessboard.avi')

ret, frame = cap.read()
if not ret:
    print("영상 파일을 열 수 없습니다.")
    cap.release()
    exit()

frame_size = (frame.shape[1], frame.shape[0])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)

    if ret_corners:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret_corners)

    cv2.imshow('Calibration', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    print("Camera Matrix (Intrinsic Parameters):")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    rmse = total_error / len(objpoints)
    print(f"\nRMSE: {rmse}")

    os.makedirs('camera/data', exist_ok=True)
    np.savez('camera/data/calibration_result.npz', mtx=mtx, dist=dist, rmse=rmse)
    print("\nCalibration result saved in camera/data/calibration_result.npz")
else:
    print("실패")
