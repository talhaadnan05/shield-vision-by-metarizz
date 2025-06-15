"""
RTSP Integration for Safe Women Application

This file shows how to add the RTSP functionality to your main application.
"""

from flask import Response, render_template, request
import cv2
import time
import numpy as np
import os
import threading
from typing import Optional, Tuple

class RTSPCameraStream:
    """
    A class to handle RTSP camera streaming using OpenCV and threading.
    """

    def __init__(self, user: str, password: str, ip: str, port: int, stream_path: str, 
                 reconnect_attempts: int = 3, reconnect_delay: float = 2.0):
        """
        Initialize the RTSPCameraStream object.

        Args:
            user (str): Username for RTSP authentication.
            password (str): Password for RTSP authentication.
            ip (str): IP address of the camera.
            port (int): Port number for the RTSP stream.
            stream_path (str): Path to the specific stream on the camera.
            reconnect_attempts (int): Number of reconnection attempts if connection fails.
            reconnect_delay (float): Delay between reconnection attempts in seconds.
        """
        self.url = f"rtsp://{user}:{password}@{ip}:{port}/{stream_path}"
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.thread = None

    def _connect(self) -> bool:
        """
        Connect to the RTSP stream.
        
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        # Set environment variables for better RTSP performance
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        
        # Create capture
        self.cap = cv2.VideoCapture(self.url)
        
        # Set buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Check if connection was successful
        if not self.cap.isOpened():
            print(f"Failed to connect to RTSP stream: {self.url}")
            return False
        
        print(f"Successfully connected to RTSP stream: {self.url}")
        return True

    def start(self) -> bool:
        """
        Start the camera stream in a separate thread.
        
        Returns:
            bool: True if the stream was started successfully, False otherwise.
        """
        # Try to connect with multiple attempts
        for attempt in range(self.reconnect_attempts):
            if self._connect():
                self.is_running = True
                self.thread = threading.Thread(target=self._update_frame)
                self.thread.daemon = True
                self.thread.start()
                return True
            
            print(f"Connection attempt {attempt+1}/{self.reconnect_attempts} failed. Retrying in {self.reconnect_delay} seconds...")
            time.sleep(self.reconnect_delay)
        
        print(f"Failed to connect to RTSP stream after {self.reconnect_attempts} attempts")
        return False

    def stop(self) -> None:
        """
        Stop the camera stream and release resources.
        """
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def _update_frame(self) -> None:
        """
        Continuously update the current frame from the camera stream.
        This method runs in a separate thread.
        """
        consecutive_failures = 0
        max_failures = 5
        
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                print("Connection lost. Attempting to reconnect...")
                if not self._connect():
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"Failed to reconnect after {max_failures} attempts. Stopping stream.")
                        self.is_running = False
                        break
                    time.sleep(self.reconnect_delay)
                    continue
                consecutive_failures = 0
            
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                print(f"Failed to read frame. Failure count: {consecutive_failures}/{max_failures}")
                
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures. Attempting to reconnect...")
                    self.cap.release()
                    self.cap = None
                    consecutive_failures = 0
                
                time.sleep(0.1)
                continue
            
            # Reset failure counter on successful frame read
            consecutive_failures = 0
            
            # Update frame count and timestamp
            self.frame_count += 1
            self.last_frame_time = time.time()
            
            # Store frame with thread safety
            with self.lock:
                self.frame = (ret, frame)
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the current frame from the camera stream.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple containing a boolean indicating
            if the frame was successfully read and the frame itself (if available).
        """
        with self.lock:
            if self.frame is not None:
                return self.frame
            return False, None
    
    def get_fps(self) -> float:
        """
        Calculate the current frames per second.
        
        Returns:
            float: The current FPS.
        """
        if self.last_frame_time == 0:
            return 0.0
        
        elapsed = time.time() - self.last_frame_time
        if elapsed > 5.0:  # If no frames for 5 seconds, return 0
            return 0.0
        
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def is_connected(self) -> bool:
        """
        Check if the stream is currently connected.
        
        Returns:
            bool: True if connected, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()

# Dictionary to store active streams
active_streams = {}

# Function to add RTSP routes to your Flask application
def add_rtsp_routes(app):
    """
    Add RTSP routes to your Flask application.
    
    Args:
        app: Your Flask application
    """
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the template file
    with open('templates/rtsp_direct.html', 'w') as f:
        f.write('''{% extends "base.html" %}

{% block title %}Direct RTSP Stream - Safe Women{% endblock %}

{% block content %}
<div class="container mt-4">
    <div class="row">
        <div class="col-md-12">
            <div class="card shadow-sm mb-4">
                <div class="card-header bg-primary text-white">
                    <h4><i class="fas fa-video"></i> Direct RTSP Stream</h4>
                </div>
                <div class="card-body">
                    <div class="row mb-4">
                        <div class="col-md-12">
                            <form id="rtsp-form" class="row g-3">
                                <div class="col-md-2">
                                    <label for="rtsp_user" class="form-label">Username</label>
                                    <input type="text" class="form-control" id="rtsp_user" name="rtsp_user" value="admin">
                                </div>
                                <div class="col-md-2">
                                    <label for="rtsp_password" class="form-label">Password</label>
                                    <input type="password" class="form-control" id="rtsp_password" name="rtsp_password" value="admin">
                                </div>
                                <div class="col-md-3">
                                    <label for="rtsp_ip" class="form-label">IP Address</label>
                                    <input type="text" class="form-control" id="rtsp_ip" name="rtsp_ip" value="192.168.1.60">
                                </div>
                                <div class="col-md-2">
                                    <label for="rtsp_port" class="form-label">Port</label>
                                    <input type="text" class="form-control" id="rtsp_port" name="rtsp_port" value="554">
                                </div>
                                <div class="col-md-2">
                                    <label for="rtsp_path" class="form-label">Stream Path</label>
                                    <input type="text" class="form-control" id="rtsp_path" name="rtsp_path" value="main">
                                </div>
                                <div class="col-md-1 d-flex align-items-end">
                                    <button type="button" id="connect-btn" class="btn btn-primary">
                                        <i class="fas fa-play"></i> Connect
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-12">
                            <div id="stream-status" class="alert alert-info">
                                Enter RTSP camera details and click Connect to start streaming.
                            </div>
                            <div id="stream-container" class="text-center">
                                <img id="stream-image" src="" class="img-fluid" style="max-height: 600px; display: none;">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card shadow-sm">
                <div class="card-header bg-info text-white">
                    <h4><i class="fas fa-info-circle"></i> About Direct RTSP Streaming</h4>
                </div>
                <div class="card-body">
                    <p>This page allows you to connect directly to RTSP cameras using their IP address and credentials.</p>
                    
                    <h5>Common RTSP Stream Paths:</h5>
                    <ul>
                        <li>Hikvision: <code>Streaming/Channels/1</code> or <code>h264/ch1/main/av_stream</code></li>
                        <li>Dahua: <code>cam/realmonitor?channel=1&subtype=0</code></li>
                        <li>Axis: <code>axis-media/media.amp</code></li>
                        <li>Generic: <code>live/main</code> or <code>stream1</code></li>
                    </ul>
                    
                    <div class="alert alert-warning">
                        <strong>Note:</strong> RTSP streams may not work due to network restrictions or firewall settings.
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    document.addEventListener('DOMContentLoaded', function() {
        const rtspForm = document.getElementById('rtsp-form');
        const connectBtn = document.getElementById('connect-btn');
        const streamStatus = document.getElementById('stream-status');
        const streamImage = document.getElementById('stream-image');
        
        // Connect button
        connectBtn.addEventListener('click', function() {
            const user = document.getElementById('rtsp_user').value;
            const password = document.getElementById('rtsp_password').value;
            const ip = document.getElementById('rtsp_ip').value;
            const port = document.getElementById('rtsp_port').value;
            const path = document.getElementById('rtsp_path').value;
            
            // Update UI
            streamStatus.textContent = 'Connecting to RTSP stream...';
            streamStatus.className = 'alert alert-info';
            streamImage.style.display = 'none';
            
            // Build the stream URL
            const streamUrl = `{{ url_for('rtsp_stream_direct') }}?user=${encodeURIComponent(user)}&password=${encodeURIComponent(password)}&ip=${encodeURIComponent(ip)}&port=${encodeURIComponent(port)}&path=${encodeURIComponent(path)}&t=${new Date().getTime()}`;
            
            // Set the image source
            streamImage.src = streamUrl;
            streamImage.style.display = 'block';
            
            // Handle image load success
            streamImage.onload = function() {
                streamStatus.textContent = 'Connected to RTSP stream';
                streamStatus.className = 'alert alert-success';
            };
            
            // Handle image load error
            streamImage.onerror = function() {
                streamStatus.textContent = 'Failed to connect to RTSP stream';
                streamStatus.className = 'alert alert-danger';
                streamImage.style.display = 'none';
            };
        });
    });
</script>
{% endblock %}''')
    
    # Add routes to the app
    @app.route('/rtsp_direct')
    def rtsp_direct():
        """RTSP Direct Stream page"""
        return render_template('rtsp_direct.html')
    
    @app.route('/rtsp_stream_direct')
    def rtsp_stream_direct():
        """Stream RTSP feed directly using the RTSPCameraStream class"""
        # Get RTSP parameters from request
        user = request.args.get('user', 'admin')
        password = request.args.get('password', 'admin')
        ip = request.args.get('ip', '192.168.1.60')
        port = int(request.args.get('port', '554'))
        path = request.args.get('path', 'main')
        
        # Generate a unique stream ID
        stream_id = f"{user}:{password}@{ip}:{port}/{path}"
        
        # Create or get stream
        if stream_id not in active_streams:
            # Create RTSP stream
            rtsp_stream = RTSPCameraStream(user, password, ip, port, path)
            
            # Start the stream
            if not rtsp_stream.start():
                # If connection fails, return an error image
                error_img = np.zeros((400, 600, 3), dtype=np.uint8)
                cv2.putText(error_img, "Error: Could not connect to RTSP stream", (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Encode error image as JPEG
                _, buffer = cv2.imencode('.jpg', error_img)
                frame_bytes = buffer.tobytes()
                
                return Response(b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n',
                               mimetype='multipart/x-mixed-replace; boundary=frame')
            
            # Store stream
            active_streams[stream_id] = rtsp_stream
        
        # Generate frames
        def generate_frames():
            try:
                while True:
                    # Check if stream is still active
                    if stream_id not in active_streams:
                        break
                    
                    # Read a frame
                    ret, frame = active_streams[stream_id].read()
                    
                    if not ret:
                        # If frame reading fails, show an error image
                        error_img = np.zeros((400, 600, 3), dtype=np.uint8)
                        cv2.putText(error_img, "Error reading frame", (50, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Encode error image as JPEG
                        _, buffer = cv2.imencode('.jpg', error_img)
                        frame_bytes = buffer.tobytes()
                    else:
                        # Add timestamp to the frame
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(frame, timestamp, (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Encode frame as JPEG
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                    
                    # Yield the frame in the response
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                    
                    # Small delay to control frame rate
                    time.sleep(0.01)
            
            except GeneratorExit:
                # Clean up when client disconnects
                if stream_id in active_streams:
                    active_streams[stream_id].stop()
                    del active_streams[stream_id]
        
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Add a link to the navigation menu
    print("RTSP routes added to the application")
    print("Don't forget to add a link to the navigation menu:")
    print("""
    <li class="nav-item">
        <a class="nav-link" href="{{ url_for('rtsp_direct') }}">
            <i class="fas fa-video"></i> RTSP Stream
        </a>
    </li>
    """)

# Example usage
if __name__ == "__main__":
    from flask import Flask
    
    # Create a simple Flask app for demonstration
    app = Flask(__name__)
    
    # Add RTSP routes to the app
    add_rtsp_routes(app)
    
    # Run the app
    app.run(host='0.0.0.0', port=5003, debug=True, threaded=True)
