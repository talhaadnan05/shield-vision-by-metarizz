from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_mail import Mail
from flask_mail import Message

bp = Blueprint('routes', __name__)
mail = Mail()

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/contact', methods=['GET', 'POST'])
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
                recipients=['adityamanapure22@gmail.com'],
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
            
        return redirect(url_for('routes.contact'))
        
    return render_template('contact.html')

@bp.route('/privacy')
def privacy():
    return render_template('privacy_policy.html')

@bp.route('/terms')
def terms():
    return render_template('terms_of_service.html')