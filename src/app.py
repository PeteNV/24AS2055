import cv2
import numpy as np
from linebot import LineBotApi, LineBotSdkDeprecatedIn30
from linebot.models import TextSendMessage, ImageSendMessage
import requests
import warnings

warnings.filterwarnings("ignore", category=LineBotSdkDeprecatedIn30)

# Set your LINE Channel Access Token and Secret
channel_access_token = (YOUR LINE CHANNEL ACCESS TOKEN HERE)

channel_secret = 'YOUR LINE CHANNEL SECRET HERE'

# Replace with your actual Line user ID
user_id = 'YOUR LINE USER ID PROVIDED IN THE DEVELOPER CONSOLE'

# Create a LineBotApi instance
line_bot_api = LineBotApi(channel_access_token)

# Load YOLOv3 weights and configuration
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO class names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Open a connection to your camera (0 represents the default camera, change it if needed)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Get image shape
    height, width, _ = frame.shape

    # Create a copy of the frame to keep the green rectangles from the camera preview
    frame_with_rectangles = frame.copy()

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass
    outputs = net.forward(output_layer_names)

    # Lists to store detected objects' information
    boxes = []
    confidences = []
    class_ids = []

    # Threshold for confidence in object detection
    conf_threshold = 0.5

    # Threshold for non-maximum suppression
    nms_threshold = 0.4

    # Flag to track knife detection
    knife_detected = False

    # Iterate through each output layer
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Check if a knife is detected
                if class_id == 43:  # Class ID 43 corresponds to "knife"
                    knife_detected = True

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw bounding boxes on the frame with rectangles, *commented out in this file, feel free to remove the comment tags!*
    # for i in indices:
        # index = i
        # box = boxes[index]
        # x, y, w, h = box
        # class_id = class_ids[index]

        # # confidence = confidences[index]
        # cv2.rectangle(frame_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(frame_with_rectangles, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame with rectangles to a temporary file
    cv2.imwrite('temp.jpg', frame_with_rectangles)

    # Send a Line notification if a knife is detected
    if knife_detected:
        print("Knife detected")

        # Upload the image to Imgur and get the image URL
        imgur_client_id = 'YOUR IMGUR CLIENT ID'
        response = requests.post(
            'https://api.imgur.com/3/upload',
            headers={'Authorization': f'Client-ID {imgur_client_id}'},
            files={'image': open('temp.jpg', 'rb')}
        )
        imgur_response = response.json()
        imgur_link = imgur_response.get('data', {}).get('link', '')

        # Send Line messages
        text_message = TextSendMessage(text='Knife detected!')
        line_bot_api.push_message(user_id, text_message)

        image_message = ImageSendMessage(
            original_content_url=imgur_link,
            preview_image_url=imgur_link
        )
        line_bot_api.push_message(user_id, image_message)

    # Display the result
    cv2.imshow('Detection', frame_with_rectangles)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
