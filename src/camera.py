from flask import Blueprint, render_template, request, redirect, url_for, session, Response, flash
from functools import wraps
from .database import get_connection, return_connection
from .utils import generate_frames
import sqlite3

bp = Blueprint('camera', __name__)

# Dictionary to store the camera feeds
camera_feeds = {}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@bp.route('/home', methods=['GET', 'POST'])
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
            
            # Cache in memory
            camera_feeds[feed_name] = feed_url
            flash(f'Camera feed "{feed_name}" added successfully', 'success')
        except sqlite3.IntegrityError:
            flash(f'A camera with name "{feed_name}" already exists', 'error')
        except Exception as e:
            flash(f'Database error: {str(e)}', 'error')
        
        return redirect(url_for('camera.home'))
    
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
            camera_feeds[row['feed_name']] = row['feed_url']
    except Exception as e:
        flash(f'Error loading camera feeds: {str(e)}', 'error')
    
    return render_template('home.html', camera_feeds=camera_feeds_for_user, role=session.get('role', 'user'))

@bp.route('/delete_camera/<feed_name>', methods=['POST'])
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
        
        # Remove from in-memory cache
        if feed_name in camera_feeds:
            del camera_feeds[feed_name]
        
        flash(f'Camera feed "{feed_name}" deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting camera feed: {str(e)}', 'error')
    
    return redirect(url_for('camera.live_feed'))

@bp.route('/video_feed/<feed_name>')
@login_required
def video_feed(feed_name):
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
            # Cache feed URL
            camera_feeds[feed_name] = feed['feed_url']
            
            return Response(generate_frames(feed_name, feed['emergency_contact']),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {str(e)}")
    
    return "Feed not found", 404

@bp.route('/live_feed')
@login_required
def live_feed():
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
            camera_feeds[row['feed_name']] = row['feed_url']
    except Exception as e:
        flash(f'Error loading camera feeds: {str(e)}', 'error')
    
    return render_template('live_feed.html', camera_feeds=camera_feeds_for_user, role=session.get('role', 'user'))