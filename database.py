"""
Database Management System
SQLite-based persistent storage for users, decisions, and chat history
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import hashlib
import os

DATABASE_PATH = "decision_system.db"


class DatabaseManager:
    """Manages all database operations for the decision system"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    goal TEXT,
                    constraints TEXT,
                    alternatives TEXT,
                    final_choice TEXT,
                    reasoning TEXT,
                    expected_outcome TEXT,
                    memory_layer TEXT DEFAULT 'private',
                    tags TEXT,
                    reflection TEXT,
                    outcome_status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # Chat history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    decision_id TEXT,
                    chat_type TEXT,
                    user_message TEXT,
                    ai_response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_visible_to_user INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (decision_id) REFERENCES decisions(id) ON DELETE CASCADE
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE NOT NULL,
                    share_data_with_ai INTEGER DEFAULT 0,
                    view_chat_history INTEGER DEFAULT 1,
                    language TEXT DEFAULT 'en',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            conn.commit()
    
    # User Management
    def create_user(self, email: str, password: str, full_name: str = "") -> bool:
        """Create a new user account"""
        try:
            password_hash = self._hash_password(password)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (email, password_hash, full_name)
                    VALUES (?, ?, ?)
                """, (email, password_hash, full_name))
                
                # Create user preferences
                user_id = cursor.lastrowid
                cursor.execute("""
                    INSERT INTO user_preferences (user_id)
                    VALUES (?)
                """, (user_id,))
                
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False  # Email already exists
    
    def authenticate_user(self, email: str, password: str) -> Optional[int]:
        """Authenticate user and return user_id if successful"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
            result = cursor.fetchone()
            
            if result and self._verify_password(password, result['password_hash']):
                return result['id']
        return None
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, email, full_name, created_at FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, email, full_name, created_at FROM users WHERE email = ?", (email,))
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def update_user_password(self, user_id: int, new_password: str) -> bool:
        """Update user password"""
        try:
            password_hash = self._hash_password(new_password)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (password_hash, user_id))
                conn.commit()
                return True
        except Exception:
            return False
    
    # Decision Management
    def save_decision(self, user_id: int, decision_data: Dict) -> str:
        """Save or update a decision"""
        decision_id = decision_data.get('id') or self._generate_id()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if decision exists
            cursor.execute("SELECT id FROM decisions WHERE id = ? AND user_id = ?", (decision_id, user_id))
            exists = cursor.fetchone() is not None
            
            constraints_json = json.dumps(decision_data.get('constraints', []))
            alternatives_json = json.dumps(decision_data.get('alternatives', []))
            tags_json = json.dumps(decision_data.get('tags', []))
            
            if exists:
                cursor.execute("""
                    UPDATE decisions SET
                        title = ?, description = ?, goal = ?, constraints = ?,
                        alternatives = ?, final_choice = ?, reasoning = ?,
                        expected_outcome = ?, memory_layer = ?, tags = ?,
                        reflection = ?, outcome_status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND user_id = ?
                """, (
                    decision_data.get('title'),
                    decision_data.get('description'),
                    decision_data.get('goal'),
                    constraints_json,
                    alternatives_json,
                    decision_data.get('final_choice'),
                    decision_data.get('reasoning'),
                    decision_data.get('expected_outcome'),
                    decision_data.get('memory_layer', 'private'),
                    tags_json,
                    decision_data.get('reflection'),
                    decision_data.get('outcome_status'),
                    decision_id,
                    user_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO decisions
                    (id, user_id, title, description, goal, constraints, alternatives,
                     final_choice, reasoning, expected_outcome, memory_layer, tags, reflection, outcome_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision_id,
                    user_id,
                    decision_data.get('title'),
                    decision_data.get('description'),
                    decision_data.get('goal'),
                    constraints_json,
                    alternatives_json,
                    decision_data.get('final_choice'),
                    decision_data.get('reasoning'),
                    decision_data.get('expected_outcome'),
                    decision_data.get('memory_layer', 'private'),
                    tags_json,
                    decision_data.get('reflection'),
                    decision_data.get('outcome_status')
                ))
            
            conn.commit()
        return decision_id
    
    def get_decision(self, user_id: int, decision_id: str) -> Optional[Dict]:
        """Get a specific decision"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM decisions WHERE id = ? AND user_id = ?
            """, (decision_id, user_id))
            result = cursor.fetchone()
            
            if result:
                data = dict(result)
                data['constraints'] = json.loads(data['constraints']) if data['constraints'] else []
                data['alternatives'] = json.loads(data['alternatives']) if data['alternatives'] else []
                data['tags'] = json.loads(data['tags']) if data['tags'] else []
                return data
        return None
    
    def get_user_decisions(self, user_id: int, limit: int = None) -> List[Dict]:
        """Get all decisions for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM decisions WHERE user_id = ? ORDER BY created_at DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (user_id,))
            results = cursor.fetchall()
            
            decisions = []
            for row in results:
                data = dict(row)
                data['constraints'] = json.loads(data['constraints']) if data['constraints'] else []
                data['alternatives'] = json.loads(data['alternatives']) if data['alternatives'] else []
                data['tags'] = json.loads(data['tags']) if data['tags'] else []
                decisions.append(data)
            
            return decisions
    
    def delete_decision(self, user_id: int, decision_id: str) -> bool:
        """Delete a decision"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM decisions WHERE id = ? AND user_id = ?", (decision_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
    
    # Chat History Management
    def save_chat_message(self, user_id: int, user_message: str, ai_response: str,
                         chat_type: str = "decision_recording", decision_id: str = None,
                         is_visible: bool = True) -> int:
        """Save a chat exchange"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history
                (user_id, decision_id, chat_type, user_message, ai_response, is_visible_to_user)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, decision_id, chat_type, user_message, ai_response, int(is_visible)))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_chat_history(self, user_id: int, decision_id: str = None,
                        chat_type: str = None, include_hidden: bool = False) -> List[Dict]:
        """Get chat history for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM chat_history WHERE user_id = ?"
            params = [user_id]
            
            if decision_id:
                query += " AND decision_id = ?"
                params.append(decision_id)
            
            if chat_type:
                query += " AND chat_type = ?"
                params.append(chat_type)
            
            if not include_hidden:
                query += " AND is_visible_to_user = 1"
            
            query += " ORDER BY timestamp ASC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
    
    def get_chat_history_for_decision(self, user_id: int, decision_id: str) -> List[Dict]:
        """Get chat history for a specific decision"""
        return self.get_chat_history(user_id, decision_id=decision_id, include_hidden=True)
    
    # User Preferences
    def get_user_preferences(self, user_id: int) -> Dict:
        """Get user preferences"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            
            if result:
                data = dict(result)
                data['share_data_with_ai'] = bool(data['share_data_with_ai'])
                data['view_chat_history'] = bool(data['view_chat_history'])
                return data
        
        return {'share_data_with_ai': False, 'view_chat_history': True}
    
    def update_user_preferences(self, user_id: int, preferences: Dict) -> bool:
        """Update user preferences"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            share_data = int(preferences.get('share_data_with_ai', False))
            view_history = int(preferences.get('view_chat_history', True))
            
            cursor.execute("""
                UPDATE user_preferences SET
                    share_data_with_ai = ?, view_chat_history = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (share_data, view_history, user_id))
            
            conn.commit()
            return cursor.rowcount > 0
    
    # Helper methods
    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def _verify_password(password: str, hash_value: str) -> bool:
        """Verify password against hash"""
        return hashlib.sha256(password.encode()).hexdigest() == hash_value
    
    @staticmethod
    def _generate_id() -> str:
        """Generate a unique ID"""
        import uuid
        return str(uuid.uuid4())


# Initialize global database instance
db = DatabaseManager()
