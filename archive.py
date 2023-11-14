import cv2
import requests
import json

# Define Line Bot credentials
LINE_CHANNEL_ID = 'channel_id'
LINE_CHANNEL_SECRET = 'channel_secret'
LINE_ACCESS_TOKEN = 'access_token'

# YOLOv3 configuration
yolo_config = 'yolov3.cfg'
yolo_weights = 'yolov3.weights'
yolo_classes = 'coco.names'

# Load YOLO model
net = cv2.dnn.readNet(yolo_weights, yolo_config)
with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

# Function to send a Line message
def send_line_message(message):
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LINE_ACCESS_TOKEN}'
    }
    data = {
        'to': '<user_id>',  # Replace with the Line user ID to send messages to
        'messages': [{'type': 'text', 'text': message}]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.status_code, response.content)

# Capture video from a camera (or process images)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Perform YOLOv3 object detection on the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    # Loop through detected objects
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'knife':
                # Knife detected, send a Line message
                send_line_message("Knife detected!")

    cv2.imshow('Knife Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
