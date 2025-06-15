from flask import Flask
from flask_mail import Mail
from .config import Config
from .database import init_db
from .routes import bp as routes_bp
from .auth import bp as auth_bp
from .camera import bp as camera_bp
from .alerts import bp as alerts_bp
from .admin import bp as admin_bp

mail = Mail()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    mail.init_app(app)

    # Initialize database
    init_db()

    # Register blueprints
    app.register_blueprint(routes_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(camera_bp)
    app.register_blueprint(alerts_bp)
    app.register_blueprint(admin_bp)

    return app