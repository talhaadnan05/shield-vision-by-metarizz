from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from .database import get_connection, return_connection
from .verification import generate_verification_code, send_verification_sms, verify_sms_code
import re

bp = Blueprint('alerts', __name__)

# Global alerts list
alerts = []

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@bp.route('/emergency_contact', methods=['GET', 'POST'])
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
                
                # Send verification code
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

@bp.route('/verify_emergency_contact', methods=['GET', 'POST'])
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
            
            # Verify the code
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

@bp.route('/alerts')
@login_required
def get_alerts():
    return {"alerts": alerts[-10:]}  # Return last 10 alerts

@bp.route('/clear_alerts')
@login_required
def clear_alerts():
    global alerts
    alerts = []
    return redirect(url_for('camera.live_feed'))