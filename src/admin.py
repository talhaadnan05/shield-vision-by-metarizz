from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from functools import wraps
from .database import get_connection, return_connection

bp = Blueprint('admin', __name__)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('auth.login'))
        
        try:
            connection = get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT role FROM users WHERE username = ?", (session['username'],))
            user = cursor.fetchone()
            cursor.close()
            return_connection(connection)
            
            if not user or user['role'] != 'admin':
                flash('Admin access required for this page', 'error')
                return redirect(url_for('camera.home'))
                
            return f(*args, **kwargs)
        except Exception as e:
            flash(f'Database error: {str(e)}', 'error')
            return redirect(url_for('camera.home'))
            
    return decorated_function

@bp.route('/users')
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
        return redirect(url_for('camera.home'))

@bp.route('/users/delete/<username>', methods=['POST'])
@admin_required
def delete_user(username):
    if username == 'admin':
        flash('Cannot delete the admin account', 'error')
        return redirect(url_for('admin.manage_users'))
        
    if username == session['username']:
        flash('Cannot delete your own account', 'error')
        return redirect(url_for('admin.manage_users'))
    
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
    
    return redirect(url_for('admin.manage_users'))

@bp.route('/users/change_role/<username>', methods=['POST'])
@admin_required
def change_role(username):
    if username == 'admin':
        flash('Cannot change the role of the admin account', 'error')
        return redirect(url_for('admin.manage_users'))
        
    role = request.form.get('role')
    if role not in ['admin', 'user']:
        flash('Invalid role', 'error')
        return redirect(url_for('admin.manage_users'))
    
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
    
    return redirect(url_for('admin.manage_users'))