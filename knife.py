import cv2
import numpy as np
from linebot import LineBotApi
from linebot.models import TextSendMessage

# LINE Messaging API configuration
LINE_CHANNEL_ACCESS_TOKEN = 'YOUR_CHANNEL_ACCESS_TOKEN'
LINE_USER_ID = 'USER_ID_TO_SEND_NOTIFICATION'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)

# Load object classes from coco.names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# YOLOv3 configuration
yolo_net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = yolo_net.getUnconnectedOutLayersNames()

def send_line_message(message):
    line_bot_api.push_message(LINE_USER_ID, TextSendMessage(text=message))

def detect_knife(frame):
    height, width, channels = frame.shape

    # Prepare frame for YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(layer_names)

    confidences = []
    class_ids = []

    # Process YOLOv3 output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'knife':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression
    indices = cv2.dnn.NMSBoxes([[center_x, center_y, width, height]], confidences, 0.5, 0.4)

    if len(indices) > 0:
        return True
    else:
        return False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if ret:
        # Detect knife
        if detect_knife(frame):
            send_line_message("Knife detected!")

        # Display the camera feed
        cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
