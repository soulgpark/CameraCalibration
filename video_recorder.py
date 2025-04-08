import cv2
import os

save_path = 'camera/data'
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
recording = False

print("스페이스바: 녹화 시작/중지 | ESC: 종료")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Chessboard Recorder', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
      
    elif key == 32:
        recording = not recording
        if recording:
            print("녹화 시작")
            out = cv2.VideoWriter(os.path.join(save_path, 'chessboard.avi'), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        else:
            print("녹화 중지")
            if out is not None:
                out.release()
                out = None

    if recording and out is not None:
        out.write(frame)

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
