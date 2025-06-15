from flask import Flask, render_template, request, redirect, url_for, session, Response, flash, jsonify
import sqlite3
import os
import torch
import cv2
import numpy as np
from functools import wraps
try:
    from mobilenet_lstm_detector import CombinedDetector
    print("🔍 Using full TensorFlow-based detector")
except ImportError:
    from mobilenet_lstm_detector_simple import CombinedDetector
    print("⚠️  Using simplified detector (TensorFlow not available)")

# Import audio capture
try:
    from audio_capture import initialize_audio, get_audio_analysis, cleanup_audio
    AUDIO_AVAILABLE = True
    print("🎤 Audio capture module loaded")
except ImportError as e:
    AUDIO_AVAILABLE = False
    print(f"⚠️  Audio capture not available: {e}")
import threading
import time
import hashlib
import re
import requests
from flask_mail import Mail, Message
import logging
from os import environ
import dotenv
from urllib.parse import urlparse

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'shield_vision_secret_key')

# SQLite Configuration
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shield_vision.db')

def get_db():
    """Get database connection"""
    db = sqlite3.connect(DATABASE_PATH)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize database tables"""
    try:
        db = get_db()
        cursor = db.cursor()

        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            emergency_contact TEXT,
            emergency_contact_verified INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create camera_feeds table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS camera_feeds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            feed_name TEXT NOT NULL,
            feed_url TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE,
            UNIQUE(username, feed_name)
        )
        ''')

        # Check if admin user exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        admin_count = cursor.fetchone()[0]

        # Create admin user if it doesn't exist
        if admin_count == 0:
            admin_password = hashlib.sha256('password'.encode()).hexdigest()
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ('admin', admin_password, 'admin')
            )
            print("✅ Default admin user created")

        db.commit()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise
    finally:
        db.close()

# Initialize database
init_db()

# Update all database operations to use SQLite
def get_connection():
    """Get database connection (compatibility function)"""
    return get_db()

def return_connection(conn):
    """Close database connection (compatibility function)"""
    conn.close()

# Dictionary to store the camera feeds and their detection status
camera_feeds = {}
detection_status = {}
alerts = []

# Load MobileNet LSTM and Shield Vision Audio models
try:
    detector = CombinedDetector()
    print(f"🔍 MobileNet LSTM and Shield Vision Audio detectors initialized")
except Exception as e:
    print(f"❌ Error loading detection models: {e}")
    detector = None

# Initialize audio capture
audio_initialized = False
if AUDIO_AVAILABLE:
    try:
        audio_initialized = initialize_audio()
        if audio_initialized:
            print("🎤 Audio capture initialized successfully")
        else:
            print("⚠️  Audio capture initialization failed")
    except Exception as e:
        print(f"⚠️  Audio capture error: {e}")
        audio_initialized = False
else:
    print("⚠️  Audio capture not available")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))

        try:
            connection = get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT role FROM users WHERE username = ?", (session['username'],))
            user = cursor.fetchone()
            cursor.close()
            return_connection(connection)

            if not user or user['role'] != 'admin':
                flash('Admin access required for this page', 'error')
                return redirect(url_for('home'))

            return f(*args, **kwargs)
        except Exception as e:
            flash(f'Database error: {str(e)}', 'error')
            return redirect(url_for('home'))

    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

            if user and user['password'] == hashlib.sha256(password.encode()).hexdigest():
                session['username'] = username
                session['role'] = user['role']
                return redirect(url_for('home'))
            else:
                error = 'Invalid credentials. Please try again.'
        except Exception as e:
            error = f'Database error: {str(e)}'
        finally:
            db.close()

    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Form validation
        if len(username) < 4:
            error = 'Username must be at least 4 characters long.'
        elif not re.match(r'^[a-zA-Z0-9_]+$', username):
            error = 'Username can only contain letters, numbers, and underscores.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters long.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        else:
            try:
                connection = get_connection()
                cursor = connection.cursor()

                # Check if username exists
                cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
                user_exists = cursor.fetchone()

                if user_exists:
                    error = 'Username already exists. Please choose another one.'
                else:
                    # Add new user
                    hashed_password = hashlib.sha256(password.encode()).hexdigest()
                    cursor.execute(
                        "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                        (username, hashed_password, 'user')
                    )
                    connection.commit()

                    # Set success message
                    flash('Account created successfully! You can now log in.', 'success')

                    cursor.close()
                    return_connection(connection)
                    return redirect(url_for('login'))

                cursor.close()
                return_connection(connection)
            except Exception as e:
                error = f'Database error: {str(e)}'

    return render_template('signup.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('index'))

@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        feed_name = request.form['feed_name']
        feed_url = request.form['feed_url']

        try:
            connection = get_connection()
            cursor = connection.cursor()

            # Store the camera feed in the database
            cursor.execute(
                "INSERT INTO camera_feeds (username, feed_name, feed_url) VALUES (?, ?, ?)",
                (session['username'], feed_name, feed_url)
            )
            connection.commit()
            cursor.close()
            return_connection(connection)

            flash(f'Camera feed "{feed_name}" added successfully', 'success')
        except sqlite3.IntegrityError as err:
            flash(f'A camera with name "{feed_name}" already exists', 'error')
        except Exception as e:
            flash(f'Database error: {str(e)}', 'error')

        return redirect(url_for('home'))

    # Get user's camera feeds from database
    camera_feeds_for_user = {}
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT feed_name, feed_url FROM camera_feeds WHERE username = ?",
            (session['username'],)
        )
        results = cursor.fetchall()
        cursor.close()
        return_connection(connection)

        for row in results:
            camera_feeds_for_user[row['feed_name']] = row['feed_url']
    except Exception as e:
        flash(f'Error loading camera feeds: {str(e)}', 'error')

    return render_template('home.html', camera_feeds=camera_feeds_for_user, role=session.get('role', 'user'), detector=detector)

@app.route('/preview_feed', methods=['POST'])
@login_required
def preview_feed():
    """Preview a camera feed before adding it to the database"""
    feed_url = request.form.get('feed_url', '')

    if not feed_url:
        return jsonify({'success': False, 'message': 'No feed URL provided'})

    # Check if the feed URL is valid
    try:
        # For webcam (0)
        if feed_url == '0':
            return jsonify({'success': True, 'message': 'Webcam feed is valid'})

        # For RTSP URLs
        if feed_url.startswith('rtsp://'):
            # Try to open the feed
            cap = cv2.VideoCapture(feed_url)
            if not cap.isOpened():
                return jsonify({'success': False, 'message': 'Could not connect to RTSP feed'})

            # Read a frame to verify the feed is working
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return jsonify({'success': False, 'message': 'Could not read from RTSP feed'})

            # Feed is valid
            cap.release()
            return jsonify({'success': True, 'message': 'RTSP feed is valid'})

        # For file paths
        if os.path.exists(feed_url):
            cap = cv2.VideoCapture(feed_url)
            if not cap.isOpened():
                return jsonify({'success': False, 'message': 'Could not open video file'})

            cap.release()
            return jsonify({'success': True, 'message': 'Video file is valid'})

        return jsonify({'success': False, 'message': 'Invalid feed URL format'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error validating feed: {str(e)}'})

def generate_preview_frames(feed_url):
    """Generate frames for preview"""
    print(f"Starting preview stream for: {feed_url}")
    try:
        # Handle different feed URL types
        if feed_url == '0':
            print("Opening webcam")
            cap = cv2.VideoCapture(0)  # Webcam
        else:
            print(f"Opening video source: {feed_url}")
            cap = cv2.VideoCapture(feed_url)

        if not cap.isOpened():
            print(f"Error: Could not open video source: {feed_url}")
            error_img = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(error_img, "Error: Could not open video source", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Encode error image as JPEG
            _, buffer = cv2.imencode('.jpg', error_img)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            return

        print(f"Video source opened successfully")

        # Generate 300 frames max for preview (about 10 seconds)
        frame_count = 0
        max_frames = 300

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"End of stream or error reading frame at count {frame_count}")
                if frame_count == 0:
                    # If we couldn't read any frames, show an error image
                    error_img = np.zeros((300, 400, 3), dtype=np.uint8)
                    cv2.putText(error_img, "No frames could be read from source", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Encode error image as JPEG
                    _, buffer = cv2.imencode('.jpg', error_img)
                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                break

            # Add frame counter to the frame
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame in the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

            frame_count += 1

            # Small delay to control frame rate
            time.sleep(0.03)  # ~30 fps

        print(f"Preview stream ended after {frame_count} frames")
        cap.release()
    except Exception as e:
        print(f"Error in preview stream: {str(e)}")
        error_img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Error: {str(e)}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Encode error image as JPEG
        _, buffer = cv2.imencode('.jpg', error_img)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/stream_preview')
@login_required
def stream_preview():
    """Stream a preview of the camera feed"""
    feed_url = request.args.get('feed_url', '')

    if not feed_url:
        return "No feed URL provided", 400

    return Response(generate_preview_frames(feed_url),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rtsp_feed')
@login_required
def rtsp_feed():
    """RTSP Feed page"""
    return render_template('rtsp_feed.html')

@app.route('/rtsp_stream')
@login_required
def rtsp_stream():
    """Stream RTSP feed"""
    url = request.args.get('url', '')

    if not url:
        return "No URL provided", 400

    return Response(generate_rtsp_stream(url),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/optimized_rtsp_feed')
@login_required
def optimized_rtsp_feed():
    """Optimized RTSP Feed page"""
    return render_template('optimized_rtsp.html')

@app.route('/optimized_rtsp_stream')
@login_required
def optimized_rtsp_stream():
    """Stream RTSP feed with optimized implementation"""
    url = request.args.get('url', '')
    use_tcp = request.args.get('tcp', 'true').lower() == 'true'
    buffer_size = int(request.args.get('buffer', '1'))

    if not url:
        return "No URL provided", 400

    return Response(generate_optimized_rtsp_stream(url, use_tcp, buffer_size),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream_page')
@login_required
def video_stream_page():
    """Video Stream page - simplified version that works with all video sources"""
    return render_template('video_stream.html')

@app.route('/video_stream')
@login_required
def video_stream():
    """Stream video from any source (webcam, file, HTTP)"""
    url = request.args.get('url', '')

    if not url:
        return "No URL provided", 400

    return Response(generate_simple_video_stream(url),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_simple_video_stream(url):
    """Generate frames from any video source - simplified version"""
    print(f"Starting video stream for: {url}")

    try:
        # Handle different URL types
        if url == '0':
            print("Opening webcam")
            cap = cv2.VideoCapture(0)  # Webcam
        else:
            print(f"Opening video source: {url}")
            cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            print(f"Error: Could not open video source: {url}")
            error_img = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(error_img, "Error: Could not open video source", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Encode error image as JPEG
            _, buffer = cv2.imencode('.jpg', error_img)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            return

        print(f"Video source opened successfully")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video properties: {width}x{height} @ {fps}fps")

        # Calculate optimal frame size for streaming (resize if too large)
        max_width = 800  # Maximum width for streaming
        if width > max_width:
            scale_factor = max_width / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resize_dimensions = (new_width, new_height)
            print(f"Resizing frames to {new_width}x{new_height} for better performance")
        else:
            resize_dimensions = None

        # Frame counter
        frame_count = 0

        # Stream indefinitely
        while True:
            # Read a frame
            ret, frame = cap.read()

            if not ret:
                print(f"End of stream or error reading frame at count {frame_count}")

                # If it's a video file, we might have reached the end
                # Try to reopen it
                if url != '0' and os.path.exists(url):
                    print("Reopening video file from the beginning")
                    cap.release()
                    cap = cv2.VideoCapture(url)
                    if not cap.isOpened():
                        break
                    continue
                else:
                    break

            # Resize frame if needed
            if resize_dimensions:
                frame = cv2.resize(frame, resize_dimensions, interpolation=cv2.INTER_AREA)

            # Add frame counter
            frame_count += 1
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame in the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

            # Small delay to control frame rate
            time.sleep(0.03)  # ~30 fps

        print(f"Video stream ended after {frame_count} frames")
        cap.release()
    except Exception as e:
        print(f"Error in video stream: {str(e)}")
        error_img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Error: {str(e)}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Encode error image as JPEG
        _, buffer = cv2.imencode('.jpg', error_img)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

def generate_rtsp_stream(url):
    """Generate frames from RTSP stream with optimized performance"""
    print(f"Starting optimized RTSP stream for: {url}")

    try:
        # Handle different URL types
        if url == '0':
            print("Opening webcam")
            cap = cv2.VideoCapture(0)  # Webcam
        else:
            print(f"Opening RTSP source: {url}")
            # Set OpenCV parameters for better RTSP performance
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

            # Use GStreamer pipeline for better performance if it's an RTSP URL
            if url.startswith('rtsp://'):
                # Optimized GStreamer pipeline for RTSP
                gst_pipeline = (
                    f'rtspsrc location={url} latency=0 ! '
                    'rtph264depay ! h264parse ! avdec_h264 ! '
                    'videoconvert ! appsink max-buffers=1 drop=true'
                )
                cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                print("Using GStreamer pipeline for RTSP")
            else:
                # Regular capture for non-RTSP sources
                cap = cv2.VideoCapture(url)
                # Set buffer size to 1 to minimize latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"Error: Could not open video source: {url}")
            error_img = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(error_img, "Error: Could not open video source", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Encode error image as JPEG with optimized parameters
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Lower quality for faster transmission
            _, buffer = cv2.imencode('.jpg', error_img, encode_params)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            return

        print(f"Video source opened successfully")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Stream properties: {width}x{height} @ {fps}fps")

        # Calculate optimal frame size for streaming (resize if too large)
        max_width = 640  # Maximum width for streaming
        if width > max_width:
            scale_factor = max_width / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resize_dimensions = (new_width, new_height)
            print(f"Resizing frames to {new_width}x{new_height} for better performance")
        else:
            resize_dimensions = None

        # Frame counter
        frame_count = 0
        start_time = time.time()
        last_frame_time = time.time()

        # JPEG encoding parameters for streaming
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Lower quality for faster transmission

        # Stream indefinitely
        while True:
            # Skip frames if processing is taking too long (to maintain real-time)
            current_time = time.time()
            time_since_last_frame = current_time - last_frame_time

            # If we're falling behind (more than 100ms since last frame), skip frame reading
            if time_since_last_frame < 0.1:  # 10 FPS minimum
                ret, frame = cap.read()
                if not ret:
                    print(f"End of stream or error reading frame at count {frame_count}")
                    if frame_count == 0:
                        # If we couldn't read any frames, show an error image
                        error_img = np.zeros((400, 600, 3), dtype=np.uint8)
                        cv2.putText(error_img, "No frames could be read from source", (50, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        # Encode error image as JPEG
                        _, buffer = cv2.imencode('.jpg', error_img, encode_params)
                        frame_bytes = buffer.tobytes()

                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

                    # Try to reconnect
                    print("Attempting to reconnect...")
                    cap.release()

                    # Recreate the capture with the same parameters
                    if url.startswith('rtsp://'):
                        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                    else:
                        cap = cv2.VideoCapture(url)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if not cap.isOpened():
                        print("Reconnection failed")
                        break
                    print("Reconnected successfully")
                    continue

                # Resize frame if needed
                if resize_dimensions:
                    frame = cv2.resize(frame, resize_dimensions, interpolation=cv2.INTER_AREA)

                # Add timestamp to the frame (smaller font for less processing)
                timestamp = time.strftime("%H:%M:%S")
                cv2.putText(frame, timestamp, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Encode frame as JPEG with optimized parameters
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                frame_bytes = buffer.tobytes()

                # Yield the frame in the response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

                # Update frame counter and timing
                frame_count += 1
                last_frame_time = time.time()

                # Calculate and log FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    current_fps = 30 / elapsed_time if elapsed_time > 0 else 0
                    print(f"Streaming at {current_fps:.2f} FPS")
                    start_time = time.time()
            else:
                # Skip frame processing if we're falling behind
                print(f"Skipping frame processing to maintain real-time (lag: {time_since_last_frame:.3f}s)")
                # Small sleep to prevent CPU overuse
                time.sleep(0.01)

        print("RTSP stream ended")
        cap.release()
    except Exception as e:
        print(f"Error in RTSP stream: {str(e)}")
        error_img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Error: {str(e)}", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Encode error image as JPEG
        _, buffer = cv2.imencode('.jpg', error_img, encode_params)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

def generate_optimized_rtsp_stream(url, use_tcp=True, buffer_size=1):
    """
    Generate frames from RTSP stream using the optimized implementation

    Args:
        url: RTSP URL to stream
        use_tcp: Whether to use TCP (more reliable) or UDP (lower latency)
        buffer_size: Size of the frame buffer (smaller = lower latency)
    """
    from optimized_rtsp import OptimizedRTSPCapture, create_error_frame

    print(f"Starting optimized RTSP stream for: {url}")
    print(f"Transport: {'TCP' if use_tcp else 'UDP'}, Buffer size: {buffer_size}")

    # Create optimized RTSP capture
    rtsp_cap = OptimizedRTSPCapture(
        url,
        buffer_size=buffer_size,
        use_tcp=use_tcp,
        verbose=True
    )

    try:
        # Start capturing
        if not rtsp_cap.start():
            print(f"Error: Could not start RTSP capture for: {url}")
            error_img = create_error_frame("Could not start RTSP capture")

            # Encode error image as JPEG
            _, buffer = cv2.imencode('.jpg', error_img)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            return

        # Calculate optimal frame size for streaming (resize if too large)
        max_width = 640  # Maximum width for streaming

        # Stream indefinitely
        while True:
            # Read a frame
            ret, frame = rtsp_cap.read()

            if not ret or frame is None:
                print("Error reading frame")
                error_img = create_error_frame("No frame available")

                # Encode error image as JPEG
                _, buffer = cv2.imencode('.jpg', error_img)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

                # Small delay before trying again
                time.sleep(0.1)
                continue

            # Resize frame if needed
            height, width = frame.shape[:2]
            if width > max_width:
                scale_factor = max_width / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Add timestamp and FPS
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            fps = rtsp_cap.get_fps()
            cv2.putText(frame, f"{timestamp} | FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame in the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

            # Small delay to control frame rate
            time.sleep(0.01)

    except Exception as e:
        print(f"Error in optimized RTSP stream: {str(e)}")
        error_img = create_error_frame(f"Error: {str(e)}")

        # Encode error image as JPEG
        _, buffer = cv2.imencode('.jpg', error_img)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    finally:
        # Clean up
        rtsp_cap.stop()

@app.route('/users')
@admin_required
def manage_users():
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT username, role, created_at FROM users")
        users = cursor.fetchall()
        cursor.close()
        return_connection(connection)
        return render_template('users.html', users=users)
    except Exception as e:
        flash(f'Database error: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/users/delete/<username>', methods=['POST'])
@admin_required
def delete_user(username):
    if username == 'admin':
        flash('Cannot delete the admin account', 'error')
        return redirect(url_for('manage_users'))

    if username == session['username']:
        flash('Cannot delete your own account', 'error')
        return redirect(url_for('manage_users'))

    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        connection.commit()
        cursor.close()
        return_connection(connection)

        flash(f'User {username} deleted successfully', 'success')
    except Exception as e:
        flash(f'Database error: {str(e)}', 'error')

    return redirect(url_for('manage_users'))

@app.route('/users/change_role/<username>', methods=['POST'])
@admin_required
def change_role(username):
    if username == 'admin':
        flash('Cannot change the role of the admin account', 'error')
        return redirect(url_for('manage_users'))

    role = request.form.get('role')
    if role not in ['admin', 'user']:
        flash('Invalid role', 'error')
        return redirect(url_for('manage_users'))

    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
        connection.commit()
        cursor.close()
        return_connection(connection)

        flash(f'Role for {username} updated to {role}', 'success')
    except Exception as e:
        flash(f'Database error: {str(e)}', 'error')

    return redirect(url_for('manage_users'))

@app.route('/delete_camera/<feed_name>', methods=['POST'])
@login_required
def delete_camera(feed_name):
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Delete the camera feed from the database
        cursor.execute(
            "DELETE FROM camera_feeds WHERE username = ? AND feed_name = ?",
            (session['username'], feed_name)
        )
        connection.commit()
        cursor.close()
        return_connection(connection)

        # Also remove from in-memory cache if present
        if feed_name in camera_feeds:
            del camera_feeds[feed_name]
        if feed_name in detection_status:
            del detection_status[feed_name]

        flash(f'Camera feed "{feed_name}" deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting camera feed: {str(e)}', 'error')

    return redirect(url_for('live_feed'))

@app.route('/emergency_contact', methods=['GET', 'POST'])
@login_required
def manage_emergency_contact():
    error = None
    success = None
    current_contact = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        if request.method == 'POST':
            phone_number = request.form['phone_number']

            # Validate phone number
            if not re.match(r'^\+?1?\d{10,14}$', phone_number):
                error = 'Invalid phone number format. Please include country code.'
            else:
                # Update emergency contact in database
                cursor.execute(
                    """UPDATE users
                    SET emergency_contact = ?,
                        emergency_contact_verified = 0
                    WHERE username = ?""",
                    (phone_number, session['username'])
                )
                connection.commit()

                # Send verification code (simulated here, you'd implement actual verification)
                verification_code = generate_verification_code()
                send_verification_sms(phone_number, verification_code)

                success = 'Emergency contact updated. Please verify your number.'

        # Retrieve current emergency contact
        cursor.execute(
            """SELECT emergency_contact, emergency_contact_verified
            FROM users WHERE username = ?""",
            (session['username'],)
        )
        user_info = cursor.fetchone()

        current_contact = {
            'number': user_info['emergency_contact'] if user_info else None,
            'verified': user_info['emergency_contact_verified'] if user_info else False
        }

        cursor.close()
        return_connection(connection)

    except Exception as e:
        error = f'Database error: {str(e)}'

    return render_template('emergency_contact.html',
                           error=error,
                           success=success,
                           current_contact=current_contact)

@app.route('/verify_emergency_contact', methods=['GET', 'POST'])
@login_required
def verify_emergency_contact():
    error = None
    success = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Retrieve current user's emergency contact
        cursor.execute(
            """SELECT emergency_contact
            FROM users WHERE username = ?""",
            (session['username'],)
        )
        user_info = cursor.fetchone()

        if request.method == 'POST':
            verification_code = request.form['verification_code']

            # Verify the code (in a real implementation, you'd check against a stored/sent code)
            if verify_sms_code(verification_code):
                # Mark contact as verified
                cursor.execute(
                    """UPDATE users
                    SET emergency_contact_verified = 1
                    WHERE username = ?""",
                    (session['username'],)
                )
                connection.commit()

                success = 'Emergency contact number verified successfully!'
            else:
                error = 'Invalid verification code. Please try again.'

        cursor.close()
        return_connection(connection)

    except Exception as e:
        error = f'Database error: {str(e)}'

    return render_template('verify_emergency_contact.html',
                           error=error,
                           success=success,
                           contact_number=user_info['emergency_contact'] if user_info else None)

def generate_verification_code():
    """Generate a 6-digit verification code"""
    import random
    return str(random.randint(100000, 999999))

def send_verification_sms(phone_number, verification_code):
    """Send SMS with verification code"""
    message = f"Your verification code is: {verification_code}"
    return send_sms_alert(phone_number, message)

def verify_sms_code(code):
    """
    Verify SMS code
    Note: In a real implementation, you'd store and check against the actual sent code
    """
    # Simulated verification - replace with actual verification logic
    return len(code) == 6 and code.isdigit()

# Modify send_sms_alert function to handle verification
def send_sms_alert(phone_number, message, server_url="https://textbelt.com/text"):
    """
    Send SMS alert using Textbelt API with improved error handling

    Args:
        phone_number (str): Phone number to send SMS to
        message (str): Message content
        server_url (str, optional): Textbelt server URL

    Returns:
        bool: True if SMS sent successfully, False otherwise
    """
    try:
        # Prepare payload for smschef API
        payload = {
            'key':'textbelt',
            'phone': phone_number,
            'message': message,
            # Optional parameters can be added here
        }

        # Send SMS via smschef API
        response = requests.post(url=server_url, data=payload, timeout=10)

        # Check response from smschef API
        response_data = response.json()

        if response.status_code == 200 and response_data.get('success', False):
            # Log successful SMS
            flash(f"SMS sent to {phone_number}")
            logging.info(f"SMS sent to {phone_number}")
            return True
        else:
            # Log failed SMS attempt
            flash(f"Failed to send SMS to {phone_number}")
            logging.warning(f"Failed to send SMS to {phone_number}. Response: {response_data}")
            return False

    except Exception as e:
        # Log any errors
        logging.error(f"Error sending SMS: {e}")
        return False

def get_location():
    try:
        response = requests.get('https://ipinfo.io')
        data = response.json()
        location = data.get('loc', 'Unknown location')
        return location
    except Exception as e:
        print(f"Error getting location: {e}")
        return 'Unknown location'

def generate_frames(feed_name, emergency_contact):
    feed_url = camera_feeds[feed_name]
    cap = cv2.VideoCapture(feed_url)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {feed_url}")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
        return

    print(f"🔍 Starting detection for feed: {feed_name}")

    message_sent = False  # Move the flag outside the loop

    while True:
        success, frame = cap.read()
        if not success:
            # Try to reopen the connection in case of network issues
            cap = cv2.VideoCapture(feed_url)
            if not cap.isOpened():
                break
            continue

        # Apply MobileNet LSTM assault detection if detector is loaded
        if detector is not None:
            detection_results = detector.detect_combined_threat(frame)
            assault_detected = detection_results['combined_threat']
            video_confidence = detection_results['video_confidence']
            audio_confidence = detection_results['audio_confidence']
            combined_confidence = detection_results['combined_confidence']

            # Draw detection information on frame
            if assault_detected:
                # Draw red border for assault detection
                height, width = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)

                # Add detection text
                cv2.putText(frame, f"THREAT DETECTED!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, f"Video: {video_confidence:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Combined: {combined_confidence:.2f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Update detection status and add alert if assault is detected
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
                message_sent = False # Reset the flag if no assault is detected


            # Add alert indicator to frame if assault detected
            if detection_status.get(feed_name, False):
                cv2.putText(frame, "ASSAULT DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Convert the frame to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    print(f"🔴 Detection stopped for feed: {feed_name}")

def generate_test_frames(source):
    """Generate frames for testing page with MobileNet LSTM detection"""
    global latest_detection_results
    print(f"🧪 Starting test detection for source: {source}")

    # Handle different source types
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open test source: {source}")
        # Create error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Error: Could not open video source", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()

        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
        return

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            print(f"End of stream or error reading frame")
            break

        frame_count += 1

        # Apply MobileNet LSTM detection if available
        if detector is not None:
            try:
                # Get real-time audio analysis if available
                audio_data = None
                real_audio_analysis = None
                if audio_initialized and AUDIO_AVAILABLE:
                    try:
                        real_audio_analysis = get_audio_analysis()
                        # Convert audio analysis to format expected by detector
                        if real_audio_analysis['status'] == 'active':
                            # Create dummy audio data for the detector
                            audio_data = np.random.random(1024) * real_audio_analysis['confidence']
                    except Exception as e:
                        print(f"Audio analysis error: {e}")

                detection_results = detector.detect_combined_threat(frame, audio_data)
                video_threat = detection_results.get('video_threat', False)
                audio_threat = detection_results.get('audio_threat', False)
                combined_threat = detection_results.get('combined_threat', False)
                video_confidence = detection_results.get('video_confidence', 0.0)
                audio_confidence = detection_results.get('audio_confidence', 0.0)
                combined_confidence = detection_results.get('combined_confidence', 0.0)

                # Override audio results with real audio analysis if available
                if audio_initialized and AUDIO_AVAILABLE and real_audio_analysis:
                    try:
                        audio_threat = real_audio_analysis['threat_detected']
                        audio_confidence = real_audio_analysis['confidence']
                        # Recalculate combined confidence
                        combined_confidence = video_confidence * 0.7 + audio_confidence * 0.3
                        combined_threat = combined_confidence > 0.6 or (video_threat and audio_threat)

                        # Update detection results
                        detection_results['audio_threat'] = audio_threat
                        detection_results['audio_confidence'] = audio_confidence
                        detection_results['combined_threat'] = combined_threat
                        detection_results['combined_confidence'] = combined_confidence
                        detection_results['audio_status'] = real_audio_analysis['status']
                        detection_results['audio_features'] = real_audio_analysis['features']

                        # Log audio activity
                        if real_audio_analysis['status'] == 'active' and real_audio_analysis['features']['volume'] > 0.1:
                            print(f"🎤 Audio detected - Volume: {real_audio_analysis['features']['volume']:.3f}, Confidence: {audio_confidence:.3f}")
                    except Exception as e:
                        print(f"Real-time audio analysis error: {e}")

                # Draw detection overlay
                height, width = frame.shape[:2]

                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Add detection status
                if combined_threat:
                    # Red border for threat
                    cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 8)
                    cv2.putText(frame, "THREAT DETECTED!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    # Green border for safe
                    cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 255, 0), 4)
                    cv2.putText(frame, "SAFE", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                # Add confidence scores
                cv2.putText(frame, f"Video: {video_confidence:.2f}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Audio: {audio_confidence:.2f}", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Combined: {combined_confidence:.2f}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Store latest results for API endpoint
                global latest_detection_results
                latest_detection_results = detection_results

            except Exception as e:
                print(f"Detection error: {e}")
                # Set default values for error case
                video_threat = False
                audio_threat = False
                combined_threat = False
                video_confidence = 0.0
                audio_confidence = 0.0
                combined_confidence = 0.0

                cv2.putText(frame, "Detection Error", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # Store error results
                latest_detection_results = {
                    'video_threat': False,
                    'audio_threat': False,
                    'combined_threat': False,
                    'video_confidence': 0.0,
                    'audio_confidence': 0.0,
                    'combined_confidence': 0.0
                }
        else:
            # No detector available
            cv2.putText(frame, "No Detector Loaded", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Control frame rate
        time.sleep(0.03)  # ~30 FPS

    cap.release()
    print(f"🔴 Test detection stopped for source: {source}")

# Global variable to store latest detection results
latest_detection_results = {
    'video_threat': False,
    'audio_threat': False,
    'combined_threat': False,
    'video_confidence': 0.0,
    'audio_confidence': 0.0,
    'combined_confidence': 0.0
}

@app.route('/video_feed/<feed_name>')
@login_required
def video_feed(feed_name):
    # Check if feed exists and belongs to user
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT feed_url, emergency_contact FROM camera_feeds JOIN users ON camera_feeds.username = users.username WHERE camera_feeds.username = ? AND feed_name = ?",
            (session['username'], feed_name)
        )
        feed = cursor.fetchone()
        cursor.close()
        return_connection(connection)

        if feed:
            # Get feed URL from database and store in memory cache for the detection process
            camera_feeds[feed_name] = feed['feed_url']

            return Response(generate_frames(feed_name,feed['emergency_contact']),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {str(e)}")

    return "Feed not found", 404

@app.route('/alerts')
@login_required
def get_alerts():
    return {"alerts": alerts[-10:]}  # Return last 10 alerts

@app.route('/live_feed')
@login_required
def live_feed():
    # Get user's camera feeds from database
    camera_feeds_for_user = {}
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT feed_name, feed_url FROM camera_feeds WHERE username = ?",
            (session['username'],)
        )
        results = cursor.fetchall()
        cursor.close()
        return_connection(connection)

        for row in results:
            camera_feeds_for_user[row['feed_name']] = row['feed_url']
            # Also cache in memory for the detection process
            camera_feeds[row['feed_name']] = row['feed_url']
    except Exception as e:
        flash(f'Error loading camera feeds: {str(e)}', 'error')

    return render_template('live_feed.html', camera_feeds=camera_feeds_for_user, role=session.get('role', 'user'))

@app.route('/clear_alerts')
@login_required
def clear_alerts():
    global alerts
    alerts = []
    return redirect(url_for('live_feed'))

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    error = None
    success = None

    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        try:
            connection = get_connection()
            cursor = connection.cursor()

            # Get current user
            cursor.execute("SELECT password FROM users WHERE username = ?", (session['username'],))
            user = cursor.fetchone()

            # Check if current password is correct
            if not user or user['password'] != hashlib.sha256(current_password.encode()).hexdigest():
                error = 'Current password is incorrect.'
            elif len(new_password) < 6:
                error = 'New password must be at least 6 characters long.'
            elif new_password != confirm_password:
                error = 'New passwords do not match.'
            else:
                # Update password
                hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                cursor.execute(
                    "UPDATE users SET password = ? WHERE username = ?",
                    (hashed_password, session['username'])
                )
                connection.commit()
                success = 'Password changed successfully!'

            cursor.close()
            return_connection(connection)
        except Exception as e:
            error = f'Database error: {str(e)}'

    return render_template('change_password.html', error=error, success=success)

dotenv.load_dotenv()
# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] =  os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] =  os.environ.get('MAIL_PASSWORD') # Use app password for Gmail
mail = Mail(app)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')

        try:
            msg = Message(
                subject=f'Contact Form: {subject}',
                sender=email,
                recipients=['shadan.anwar2005@gmail.com', 'zayyyyn.07@gmail.com'],
                body=f'''
From: {name} <{email}>

{message}
'''
            )
            mail.send(msg)
            flash('Thank you for your message. We will get back to you soon!', 'success')
        except Exception as e:
            print(f"Error sending email: {e}")
            flash('Sorry, there was an error sending your message. Please try again.', 'error')

        return redirect(url_for('contact'))

    return render_template('contact.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy_policy.html')
@app.route('/terms')
def terms():
    return render_template('terms_of_service.html')

@app.route('/testing')
@login_required
def testing_page():
    """Testing page for live video detection"""
    return render_template('testing_page.html')

@app.route('/test_video_stream')
@login_required
def test_video_stream():
    """Stream video for testing purposes"""
    source = request.args.get('source', '0')
    return Response(generate_test_frames(source),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detection_results')
@login_required
def get_detection_results():
    """Get current detection results for testing page"""
    global latest_detection_results

    # Convert numpy types to Python native types for JSON serialization
    safe_results = {
        'video_threat': bool(latest_detection_results.get('video_threat', False)),
        'audio_threat': bool(latest_detection_results.get('audio_threat', False)),
        'combined_threat': bool(latest_detection_results.get('combined_threat', False)),
        'video_confidence': float(latest_detection_results.get('video_confidence', 0.0)),
        'audio_confidence': float(latest_detection_results.get('audio_confidence', 0.0)),
        'combined_confidence': float(latest_detection_results.get('combined_confidence', 0.0))
    }

    # Add audio capture status
    if audio_initialized and AUDIO_AVAILABLE:
        try:
            audio_analysis = get_audio_analysis()
            safe_results['audio_status'] = audio_analysis['status']
            safe_results['audio_features'] = audio_analysis['features']
        except Exception as e:
            safe_results['audio_status'] = 'error'
            safe_results['audio_features'] = {}
    else:
        safe_results['audio_status'] = 'not_available'
        safe_results['audio_features'] = {}

    return jsonify(safe_results)

@app.route('/get_audio_status')
@login_required
def get_audio_status():
    """Get audio capture status"""
    status = {
        'available': AUDIO_AVAILABLE,
        'initialized': audio_initialized,
        'status': 'not_available'
    }

    if audio_initialized and AUDIO_AVAILABLE:
        try:
            audio_analysis = get_audio_analysis()
            status['status'] = audio_analysis['status']
            status['features'] = audio_analysis['features']
        except Exception as e:
            status['status'] = 'error'
            status['error'] = str(e)

    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Starting Shield Vision Application...")
    print("🔍 MobileNet LSTM + Shield Vision Audio Detection System")
    print("=" * 60)

    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    app.run(debug=True, threaded=True)
