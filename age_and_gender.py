import cv2
from ultralytics import YOLO
from deepface import DeepFace

# Load YOLO face detection model
model = YOLO("yolov8n-face.pt")   # face detection model

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # YOLO face detection
    results = model(frame)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            face = frame[y1:y2, x1:x2]

            try:
                result = DeepFace.analyze(
                    face,
                    actions=['age','gender'],
                    enforce_detection=False
                )

                age = result[0]['age']
                gender = result[0]['dominant_gender']

                label = f"{gender}, {age}"

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            except:
                pass

    cv2.imshow("Age & Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
