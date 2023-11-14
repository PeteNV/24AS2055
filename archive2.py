import cv2
import requests
import json
import time
import warnings
import numpy as np
from linebot import LineBotApi, LineBotSdkDeprecatedIn30
from linebot.models import ImageMessage, TextSendMessage


# Function to upload the image to Imgur and get the URL
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


# Function to detect black color using Imagga's Color Extraction API
def detect_black_color(image_url, api_key, api_secret):
    url = "https://api.imagga.com/v2/colors"
    response = requests.get(url, params={"image_url": image_url}, auth=(api_key, api_secret))
    data = json.loads(response.text)
    print(data)  # Print the response to examine its structure

    colors = data["result"]["colors"]["image_colors"]
    for color in colors:
        color_name = color["html_code"]
        if color_name == "#000000":  # Check if the color is black (#000000)
            return True
    return False


# Function to send the captured image to Line
def send_image_to_line(image_url):
    # Upload the image to Line server and get the media ID
    image_message = ImageMessage(original_content_url=image_url, preview_image_url=image_url)
    response = line_bot_api.push_message("U0374fd264aeadd7f483dff6ed8e568e7", image_message)  # MAYBE CHANGE?

    # Check if the message was successfully sent
    if '200' not in str(response.status_code):
        print('Failed to send image to Line.')


# Suppress LineBotSdkDeprecatedIn30 warning
warnings.filterwarnings("ignore", category=LineBotSdkDeprecatedIn30)

# Set up the Line Bot API
line_bot_api = LineBotApi(
    "LbSqSqaJ3HPNaDNGt1Nsed3SBupWtURIgeCdMcA/4oH3xMODM0NmUrz5W105tI6MBIs9jlGBLBCgoHDLQK3Gh640qp"
    "+Y6aahu37S4eRsUBkQWKPfrJL/LMWiB34F8iXdIbLLRb+107Q8LFHN0+fl3AdB04t89/1O/w1cDnyilFU=")

# Initialize the video capture object to access the live camera feed
video_capture = cv2.VideoCapture(0)  # Use '0' for the default camera

# Previous frame for motion detection
prev_frame = None

# Imgur API Client ID (Replace with your Imgur API client ID)
IMGUR_CLIENT_ID = "6d5b8aa0a550a4d"

# Main detection loop
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if prev_frame is None:
        prev_frame = frame
        continue

    # Calculate the absolute difference between the current frame and the previous frame
    frame_delta = cv2.absdiff(prev_frame, frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Convert thresh image to grayscale
    gray_thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    # Find contours of moving objects
    contours, _ = cv2.findContours(gray_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Reset motion detection for the next frame
    prev_frame = frame

    # Check if there are any moving objects (suspicious person)
    if len(contours) > 0:
        # Convert the frame to a URL-accessible image
        image_path = "/Users/petev./PycharmProjects/Main/frame.jpg"
        cv2.imwrite(image_path, frame)

        # Upload the image to Imgur and get the URL
        image_url = upload_image_to_imgur(image_path, IMGUR_CLIENT_ID)

        if image_url:
            # Detect black color in the captured frame
            suspicious_person_detected = detect_black_color(image_url, "acc_0d4632fd3eb3866",
                                                            "938370bda72e3cc671c3f293242ce75b")

            # Display the frame with a bounding box around the suspicious person
            if suspicious_person_detected:
                # Draw a red rectangle around the suspicious person
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)

                # Send the captured image to Line
                send_image_to_line(image_url)

    # Display the resulting frame
    cv2.imshow('Suspicious Person Detection', frame)

    # Add a small delay to limit the frequency of frame capturing
    # Adjust the delay time as needed (e.g., 0.1 seconds)
    # This will reduce the CPU usage and the rapid capturing of frames
    time.sleep(0.1)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video_capture.release()
cv2.destroyAllWindows()
