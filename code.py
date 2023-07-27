import cv2
import requests
import json
from linebot import LineBotApi
from linebot.models import ImageMessage, TextSendMessage


# Function to detect black color using Imagga's Color Extraction API
def detect_black_color(image_url, api_key, api_secret):
    url = "https://api.imagga.com/v2/colors"
    response = requests.get(url, params={"image_url": image_url}, auth=(api_key, api_secret))
    data = json.loads(response.text)
    colors = data["result"]["colors"]
    for color in colors:
        if color["closest_palette_color"] == "black":
            return True
    return False


# Function to send the captured image to Line
def send_image_to_line(image_path):
    # Upload the image to Line server and get the media ID
    image_message = ImageMessage(original_content_url=image_path, preview_image_url=image_path)
    response = line_bot_api.push_message(YOUR_LINE_USER_ID, image_message)

    # Check if the message was successfully sent
    if '200' not in str(response.status_code):
        print('Failed to send image to Line.')


# Set up the Line Bot API
line_bot_api = LineBotApi("YOUR_CHANNEL_ACCESS_TOKEN")

# Initialize the video capture object to access the live camera feed
video_capture = cv2.VideoCapture(0)  # Use '0' for the default camera

# Previous frame for motion detection
prev_frame = None

# Main detection loop
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve motion detection
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray_frame
        continue

    # Calculate the absolute difference between the current frame and the previous frame
    frame_delta = cv2.absdiff(prev_frame, gray_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of moving objects
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Reset motion detection for the next frame
    prev_frame = gray_frame

    # Check if there are any moving objects (suspicious person)
    if len(contours) > 0:
        # Convert the frame to a URL-accessible image
        image_path = "frame.jpg"
        cv2.imwrite(image_path, frame)
        image_url = "file://" + image_path

        # Detect black color in the captured frame
        suspicious_person_detected = detect_black_color(image_url, YOUR_API_KEY, YOUR_API_SECRET)

        # Display the frame with a bounding box around the suspicious person
        if suspicious_person_detected:
            # Draw a red rectangle around the suspicious person
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)

            # Send the captured image to Line
            send_image_to_line(image_path)

    # Display the resulting frame
    cv2.imshow('Suspicious Person Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video_capture.release()
cv2.destroyAllWindows()

# THIS CODE IS THE PROPERTY OF THE BERMUDA TRIANGLE
# NATTANAN VIMUKTANAN M.205 NO.1
# KRITTAPHAT TRAKULHTONGCHAROEN M.205 NO.12
# THITIKAN SINPRASONG M.205 NO.15

# PLEASE USE THE SOFTWARE RESPONSIBLY
