"""Setup database for Product Review Analyzer."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager
from src.core.config_loader import ConfigLoader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Setup database with schema."""
    
    print("\n" + "="*60)
    print("🗄️  Product Review Analyzer - Database Setup")
    print("="*60 + "\n")
    
    # Load config
    config_loader = ConfigLoader()
    app_config = config_loader.load_app_config()
    
    db_path = app_config['database']['path']
    
    print(f"Database path: {db_path}\n")
    
    # Check if database exists
    db_file = Path(db_path)
    if db_file.exists():
        response = input("⚠️  Database already exists. Overwrite? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Setup cancelled.")
            return
        
        # Backup old database
        backup_path = db_file.with_suffix('.db.backup')
        db_file.rename(backup_path)
        print(f"✅ Old database backed up to: {backup_path}\n")
    
    # Create database manager
    db_manager = DatabaseManager(db_path)
    
    # Load and execute schema
    schema_file = Path(__file__).parent.parent / "src" / "database" / "schema.sql"
    
    if schema_file.exists():
        print("📝 Loading schema from SQL file...")
        db_manager.load_schema_from_file(str(schema_file))
    else:
        print("📝 Creating tables from ORM models...")
        db_manager.create_tables()
    
    print("\n✅ Database setup complete!\n")
    
    # Print stats
    db_manager.print_stats()
    
    # Initialize with default category
    print("📦 Initializing with Electronics category...")
    
    from src.database.models import Category
    
    with db_manager.get_session() as session:
        # Check if Electronics category exists
        electronics = session.query(Category).filter_by(name='Electronics').first()
        
        if not electronics:
            electronics = Category(
                name='Electronics',
                amazon_id='Electronics',
                config_path='config/categories/electronics.yaml'
            )
            session.add(electronics)
            session.commit()
            print("✅ Electronics category added")
        else:
            print("ℹ️  Electronics category already exists")
    
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Download data: python scripts/download_data.py --category electronics")
    print("  2. Parse data: python scripts/parse_data.py --category electronics")
    print("  3. Run analysis: python scripts/run_analysis.py --category electronics")
    print("  4. Launch UI: streamlit run src/ui/app.py")
    print("\n")


if __name__ == "__main__":
    main()
