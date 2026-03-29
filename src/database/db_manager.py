"""Database manager for handling database operations."""

import os
import sqlite3
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, db_path: str = None, echo: bool = False):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
            echo: Whether to echo SQL statements (for debugging)
        """
        if db_path is None:
            db_path = "./data/processed/reviews.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(
            db_url,
            echo=echo,
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            },
            poolclass=StaticPool
        )
        
        # Enable foreign keys for SQLite
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            cursor.close()
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all tables defined in models."""
        Base.metadata.create_all(bind=self.engine)
        print(f"✅ Database tables created at: {self.db_path}")
    
    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)
        print(f"⚠️ All tables dropped from: {self.db_path}")
    
    def reset_database(self):
        """Drop and recreate all tables."""
        self.drop_tables()
        self.create_tables()
        print(f"🔄 Database reset complete: {self.db_path}")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db_manager.get_session() as session:
                session.query(Product).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def execute_raw_sql(self, sql: str, params: Optional[tuple] = None):
        """
        Execute raw SQL query.
        
        Args:
            sql: SQL query string
            params: Query parameters (for parameterized queries)
        
        Returns:
            Query results
        """
        with self.engine.connect() as conn:
            if params:
                result = conn.execute(sql, params)
            else:
                result = conn.execute(sql)
            conn.commit()
            return result.fetchall()
    
    def load_schema_from_file(self, schema_file: str):
        """
        Load and execute SQL schema from file.
        
        Args:
            schema_file: Path to SQL schema file
        """
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Execute schema (split by semicolons for multiple statements)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript(schema_sql)
        conn.commit()
        conn.close()
        
        print(f"✅ Schema loaded from: {schema_file}")
    
    def vacuum(self):
        """Optimize database (reclaim space, rebuild indices)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("VACUUM")
        conn.close()
        print(f"✅ Database optimized: {self.db_path}")
    
    def get_db_size(self) -> str:
        """Get database file size in human-readable format."""
        if not self.db_path.exists():
            return "0 B"
        
        size_bytes = self.db_path.stat().st_size
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.2f} TB"
    
    def get_table_counts(self) -> dict:
        """Get row counts for all tables."""
        counts = {}
        
        with self.get_session() as session:
            from .models import (
                Category, Brand, Product, Review, 
                AspectSentiment, ProductSummary, BrandSummary
            )
            
            counts['categories'] = session.query(Category).count()
            counts['brands'] = session.query(Brand).count()
            counts['products'] = session.query(Product).count()
            counts['reviews'] = session.query(Review).count()
            counts['aspect_sentiments'] = session.query(AspectSentiment).count()
            counts['product_summaries'] = session.query(ProductSummary).count()
            counts['brand_summaries'] = session.query(BrandSummary).count()
        
        return counts
    
    def print_stats(self):
        """Print database statistics."""
        print(f"\n📊 Database Statistics")
        print(f"{'='*50}")
        print(f"Path: {self.db_path}")
        print(f"Size: {self.get_db_size()}")
        print(f"\nTable Counts:")
        
        counts = self.get_table_counts()
        for table, count in counts.items():
            print(f"  {table:20s}: {count:,}")
        
        print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test database setup
    db = DatabaseManager("./data/processed/reviews.db")
    db.create_tables()
    db.print_stats()
