import cv2
import time
import requests
from ultralytics import YOLO
import logging
from .alerts import alerts
from .verification import send_sms_alert

# Load YOLO model
try:
    model_path = "models/yolo11_assault.pt"
    model = YOLO(model_path)
    print(f"🔍 YOLO model loaded from {model_path}")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

# Dictionary to store detection status
detection_status = {}

def get_location():
    try:
        response = requests.get('https://ipinfo.io')
        data = response.json()
        location = data.get('loc', 'Unknown location')
        return location
    except Exception as e:
        print(f"Error getting location: {e}")
        return 'Unknown location'

def generate_frames(feed_name, emergency_contact, camera_feeds):
    feed_url = camera_feeds[feed_name]
    cap = cv2.VideoCapture(feed_url)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {feed_url}")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
        return
    
    print(f"🔍 Starting detection for feed: {feed_name}")
    
    message_sent = False
    
    while True:
        success, frame = cap.read()
        if not success:
            cap = cv2.VideoCapture(feed_url)
            if not cap.isOpened():
                break
            continue
            
        if model is not None:
            results = model(frame)
            assault_detected = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    
                    if (label != "People" and label != "Police") and confidence > 0.5:
                        assault_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if assault_detected and not detection_status.get(feed_name, False):
                detection_status[feed_name] = True
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                location = get_location()
                alerts.append({
                    "feed_name": feed_name,
                    "timestamp": timestamp,
                    "location": location,
                    "message": f"🚨 ALERT! Assault detected in {feed_name}! Alert Sent to Emergency Contact"
                })
                
                if not message_sent:
                    send_sms_alert(emergency_contact, 
                         f"🚨 ALERT! Assault detected in {feed_name} at {timestamp} on location {location}")
                    message_sent = True
                
            elif not assault_detected:
                detection_status[feed_name] = False
                message_sent = False
            
            if detection_status.get(feed_name, False):
                cv2.putText(frame, "ASSAULT DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    print(f"🔴 Detection stopped for feed: {feed_name}")