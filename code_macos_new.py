import cv2
import requests
import json
import time
import numpy as np
from linebot import LineBotApi, LineBotSdkDeprecatedIn30
from linebot.models import ImageMessage, TextSendMessage

def upload_image_to_imgur(image_path, client_id):
    url = "https://api.imgur.com/3/upload"
    headers = {"Authorization": f"Client-ID {client_id}"}
    files = {"image": open(image_path, "rb")}
    response = requests.post(url, headers=headers, files=files)

    try:
        response_data = response.json()
        print("Imgur API Response:", response_data)  # Print the full response data for debugging
        if response.status_code == 200 and response_data.get("success"):
            image_url = response_data["data"]["link"]
            print("Image uploaded successfully:", image_url)
            return image_url
        else:
            print("Imgur API error:", response_data.get("data", {}).get("error"))
    except Exception as e:
        print("Error parsing Imgur API response:", e)

    return None

def detect_black_color(image_url, api_key, api_secret):
    url = "https://api.imagga.com/v2/colors"
    response = requests.get(url, params={"image_url": image_url}, auth=(api_key, api_secret))
    data = json.loads(response.text)

    colors = data["result"]["colors"]["image_colors"]
    for color in colors:
        color_name = color["html_code"]
        if color_name == "#000000":
            return True
    return False

def send_image_to_line(image_url):
    image_message = ImageMessage(original_content_url=image_url, preview_image_url=image_url)
    response = line_bot_api.push_message("USER_ID", image_message)  # Replace with the user's ID

    if '200' not in str(response.status_code):
        print('Failed to send image to Line.')

warnings.filterwarnings("ignore", category=LineBotSdkDeprecatedIn30)

line_bot_api = LineBotApi("LbSqSqaJ3HPNaDNGt1Nsed3SBupWtURIgeCdMcA/4oH3xMODM0NmUrz5W105tI6MBIs9jlGBLBCgoHDLQK3Gh640qp+Y6aahu37S4eRsUBkQWKPfrJL/LMWiB34F8iXdIbLLRb+107Q8LFHN0+fl3AdB04t89/1O/w1cDnyilFU=")

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

video_capture = cv2.VideoCapture(0)

prev_frame = None

while True:
    ret, frame = video_capture.read()

    if prev_frame is None:
        prev_frame = frame
        continue

    frame_delta = cv2.absdiff(prev_frame, frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    gray_thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    prev_frame = frame

    if len(contours) > 0:
        image_path = "frame.jpg"
        cv2.imwrite(image_path, frame)

        image_url = upload_image_to_imgur(image_path, "6d5b8aa0a550a4d")

        if image_url:
            suspicious_person_detected = detect_black_color(image_url, "acc_0d4632fd3eb3866", "938370bda72e3cc671c3f293242ce75b")

            if suspicious_person_detected:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)

                blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                layer_names = net.getUnconnectedOutLayersNames()
                detections = net.forward(layer_names)

                h, w = frame.shape[:2]

                for detection in detections:
                    for obj in detection:
                        scores = obj[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5:
                            center_x = int(obj[0] * w)
                            center_y = int(obj[1] * h)
                            width = int(obj[2] * w)
                            height = int(obj[3] * h)

                            x = int(center_x - width / 2)
                            y = int(center_y - height / 2)

                            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            cv2.putText(frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                send_image_to_line(image_url)

    cv2.imshow('Suspicious Person Detection', frame)

    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
