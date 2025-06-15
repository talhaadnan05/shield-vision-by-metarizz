import random
import requests
import logging

def generate_verification_code():
    """Generate a 6-digit verification code"""
    return str(random.randint(100000, 999999))

def send_verification_sms(phone_number, verification_code):
    """Send SMS with verification code"""
    message = f"Your verification code is: {verification_code}"
    return send_sms_alert(phone_number, message)

def verify_sms_code(code):
    """Verify SMS code (simulated)"""
    return len(code) == 6 and code.isdigit()

def send_sms_alert(phone_number, message, server_url="https://textbelt.com/text"):
    """
    Send SMS alert using Textbelt API with improved error handling
    """
    try:
        payload = {
            'key': 'textbelt',
            'phone': phone_number,
            'message': message,
        }
        
        response = requests.post(url=server_url, data=payload, timeout=10)
        response_data = response.json()
        
        if response.status_code == 200 and response_data.get('success', False):
            logging.info(f"SMS sent to {phone_number}")
            return True
        else:
            logging.warning(f"Failed to send SMS to {phone_number}. Response: {response_data}")
            return False
    
    except Exception as e:
        logging.error(f"Error sending SMS: {e}")
        return False 