"""Environment variable loader with validation."""

import os
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class EnvLoader:
    """Load and validate environment variables."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize environment loader.
        
        Args:
            env_file: Path to .env file. If None, searches for .env in project root
        """
        if env_file is None:
            # Search for .env in project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            env_file = project_root / ".env"
        
        self.env_file = Path(env_file)
        self.loaded = False
        
        # Load .env file if exists
        if self.env_file.exists():
            load_dotenv(self.env_file)
            self.loaded = True
            logger.info(f"Loaded environment from: {self.env_file}")
        else:
            logger.warning(f".env file not found at: {self.env_file}")
            logger.info("Using default values and system environment variables")
    
    def get_str(self, key: str, default: str = "") -> str:
        """Get string environment variable."""
        return os.getenv(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}, using default: {default}")
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float value for {key}: {value}, using default: {default}")
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_path(self, key: str, default: str = "") -> Path:
        """Get path environment variable."""
        value = os.getenv(key, default)
        return Path(value) if value else Path(default)
    
    def get_list(self, key: str, separator: str = ",", default: Optional[list] = None) -> list:
        """Get list environment variable (comma-separated by default)."""
        value = os.getenv(key)
        if value is None:
            return default or []
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def require(self, key: str) -> str:
        """
        Get required environment variable.
        
        Args:
            key: Environment variable name
        
        Returns:
            Environment variable value
        
        Raises:
            ValueError: If environment variable is not set
        """
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable not set: {key}")
        return value
    
    def validate(self) -> bool:
        """
        Validate environment configuration.
        
        Returns:
            True if configuration is valid
        """
        issues = []
        
        # Check database path parent exists
        db_path = self.get_path("DB_PATH", "./data/processed/reviews.db")
        if not db_path.parent.exists():
            issues.append(f"Database directory does not exist: {db_path.parent}")
        
        # Check cache directory
        cache_dir = self.get_path("CACHE_DIR", "./data/cache")
        if not cache_dir.exists():
            logger.warning(f"Cache directory does not exist, will create: {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check log directory
        log_file = self.get_path("LOG_FILE", "./logs/app.log")
        if not log_file.parent.exists():
            logger.warning(f"Log directory does not exist, will create: {log_file.parent}")
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check GPU settings
        use_gpu = self.get_bool("USE_GPU", True)
        if use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("USE_GPU=true but CUDA is not available, will use CPU")
            except ImportError:
                logger.warning("USE_GPU=true but PyTorch is not installed")
        
        # Report issues
        if issues:
            for issue in issues:
                logger.error(issue)
            return False
        
        logger.info("✅ Environment configuration validated")
        return True
    
    def print_config(self):
        """Print current configuration (excluding sensitive values)."""
        print("\n" + "="*60)
        print("🔧 Environment Configuration")
        print("="*60)
        
        # Database
        print("\n📁 Database:")
        print(f"  DB_PATH: {self.get_path('DB_PATH', './data/processed/reviews.db')}")
        print(f"  DB_TIMEOUT: {self.get_int('DB_TIMEOUT', 30)}s")
        print(f"  DB_ECHO: {self.get_bool('DB_ECHO', False)}")
        
        # GPU
        print("\n🎮 GPU Settings:")
        print(f"  USE_GPU: {self.get_bool('USE_GPU', True)}")
        print(f"  USE_FP16: {self.get_bool('USE_FP16', True)}")
        print(f"  CUDA_VISIBLE_DEVICES: {self.get_str('CUDA_VISIBLE_DEVICES', '0')}")
        
        # Model
        print("\n🤖 AI Model:")
        print(f"  MODEL_NAME: {self.get_str('MODEL_NAME', 'valhalla/distilbart-mnli-12-3')}")
        print(f"  BATCH_SIZE: {self.get_int('BATCH_SIZE', 32)}")
        
        # Processing
        print("\n⚙️  Processing:")
        print(f"  MIN_REVIEWS: {self.get_int('MIN_REVIEWS', 20)}")
        print(f"  TOP_PRODUCTS: {self.get_int('TOP_PRODUCTS', 3000)}")
        print(f"  CHECKPOINT_INTERVAL: {self.get_int('CHECKPOINT_INTERVAL', 10)}")
        
        # Logging
        print("\n📝 Logging:")
        print(f"  LOG_LEVEL: {self.get_str('LOG_LEVEL', 'INFO')}")
        print(f"  LOG_FILE: {self.get_path('LOG_FILE', './logs/app.log')}")
        print(f"  LOG_TO_CONSOLE: {self.get_bool('LOG_TO_CONSOLE', True)}")
        
        # Cache
        print("\n💾 Cache:")
        print(f"  CACHE_DIR: {self.get_path('CACHE_DIR', './data/cache')}")
        print(f"  ENABLE_CACHE: {self.get_bool('ENABLE_CACHE', True)}")
        print(f"  CACHE_SIZE_MB: {self.get_int('CACHE_SIZE_MB', 1024)} MB")
        
        # Development
        print("\n🛠️  Development:")
        print(f"  USE_SAMPLE_DATA: {self.get_bool('USE_SAMPLE_DATA', False)}")
        print(f"  SKIP_AI_ANALYSIS: {self.get_bool('SKIP_AI_ANALYSIS', False)}")
        
        print("="*60 + "\n")


# Global instance
_env_loader = None


def get_env() -> EnvLoader:
    """Get global environment loader instance."""
    global _env_loader
    if _env_loader is None:
        _env_loader = EnvLoader()
    return _env_loader


# Example usage
if __name__ == "__main__":
    env = EnvLoader()
    
    # Print configuration
    env.print_config()
    
    # Validate
    is_valid = env.validate()
    print(f"\nValidation: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    # Example access
    print(f"\nExample access:")
    print(f"  Database path: {env.get_path('DB_PATH')}")
    print(f"  Batch size: {env.get_int('BATCH_SIZE', 32)}")
    print(f"  Use GPU: {env.get_bool('USE_GPU', True)}")
