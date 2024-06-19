import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video/1.mp4")
ptime = time.time()

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)


while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame. Exiting...")
        break    

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection.score)
            # print(detection.location_data.relative_bounding_box)
            bbocC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bbocC.xmin * iw), int(bbocC.ymin * ih), \
                int(bbocC.width * iw), int(bbocC.height * ih)
            
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            
            cv2.putText(img, f"{int(detection.score[0] * 100)}%",
                        (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)
            
    cTime = time.time()
    time_diff = cTime - ptime
    
    if time_diff == 0:
        print("Time difference is too small.. Skipping FPS calculation for this frame")
    
    elif time_diff != 0:
        fps = 1 / time_diff
        cv2.putText(img, f"FPS: {int(fps)}",
                (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    
    ptime = cTime
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()