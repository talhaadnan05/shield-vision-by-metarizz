from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from .database import get_connection, return_connection
import hashlib
import re

bp = Blueprint('auth', __name__)

@bp.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            db = get_connection()
            cursor = db.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            
            if user and user['password'] == hashlib.sha256(password.encode()).hexdigest():
                session['username'] = username
                session['role'] = user['role']
                return redirect(url_for('camera.home'))
            else:
                error = 'Invalid credentials. Please try again.'
        except Exception as e:
            error = f'Database error: {str(e)}'
        finally:
            return_connection(db)
    
    return render_template('login.html', error=error)

@bp.route('/signup', methods=['GET', 'POST'])
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
                    
                    flash('Account created successfully! You can now log in.', 'success')
                    cursor.close()
                    return_connection(connection)
                    return redirect(url_for('auth.login'))
                
                cursor.close()
                return_connection(connection)
            except Exception as e:
                error = f'Database error: {str(e)}'
    
    return render_template('signup.html', error=error)

@bp.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('routes.index'))

@bp.route('/change_password', methods=['GET', 'POST'])
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