import sqlite3
import hashlib
from .config import Config

def get_db():
    """Get database connection"""
    db = sqlite3.connect(Config.DATABASE_PATH)
    db.row_factory = sqlite3.Row
    return db

def get_connection():
    """Get database connection (compatibility function)"""
    return get_db()

def return_connection(conn):
    """Close database connection (compatibility function)"""
    conn.close()

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