import cv2
import numpy as np

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
                if class_id == 43:  # Class ID ... corresponds to "knife"
                    knife_detected = True

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw bounding boxes on the frame
    for i in indices:
        index = i
        box = boxes[index]
        x, y, w, h = box
        class_id = class_ids[index]

        label = str(classes[class_id])
        confidence = confidences[index]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Print a message if a knife is detected
    if knife_detected:
        print("Knife detected")

    # Display the result
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
