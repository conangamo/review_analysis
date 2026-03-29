"""Configuration loader for category-specific settings."""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


class ConfigLoader:
    """Load and manage configuration from YAML files."""
    
    def __init__(self, config_dir: str = "./config"):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self.categories_dir = self.config_dir / "categories"
        
        # Cache loaded configs
        self._category_configs = {}
        self._app_config = None
        self._model_config = None
    
    def load_category_config(self, category_name: str) -> Dict[str, Any]:
        """
        Load configuration for a specific category.
        
        Args:
            category_name: Name of category (e.g., 'electronics')
        
        Returns:
            Dictionary containing category configuration
        """
        # Check cache first
        if category_name in self._category_configs:
            return self._category_configs[category_name]
        
        # Load from file
        config_file = self.categories_dir / f"{category_name.lower()}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Category config not found: {config_file}\n"
                f"Available categories: {self.list_available_categories()}"
            )
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Cache and return
        self._category_configs[category_name] = config
        return config
    
    def load_app_config(self) -> Dict[str, Any]:
        """Load application configuration."""
        if self._app_config is not None:
            return self._app_config
        
        config_file = self.config_dir / "app.yaml"
        with open(config_file, 'r', encoding='utf-8') as f:
            self._app_config = yaml.safe_load(f)
        
        return self._app_config
    
    def load_model_config(self) -> Dict[str, Any]:
        """Load AI model configuration."""
        if self._model_config is not None:
            return self._model_config
        
        config_file = self.config_dir / "models.yaml"
        with open(config_file, 'r', encoding='utf-8') as f:
            self._model_config = yaml.safe_load(f)
        
        return self._model_config
    
    def list_available_categories(self) -> List[str]:
        """List all available category configurations."""
        categories = []
        for file in self.categories_dir.glob("*.yaml"):
            if file.stem != "_template":
                categories.append(file.stem)
        return sorted(categories)
    
    def get_aspects(self, category_name: str, tier: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get aspects for a category, optionally filtered by tier.
        
        Args:
            category_name: Category name
            tier: Optional tier filter (1, 2, or 3)
        
        Returns:
            List of aspect configurations
        """
        config = self.load_category_config(category_name)
        aspects = []
        
        # Collect aspects from all tiers
        for tier_name, tier_aspects in config['aspects'].items():
            tier_num = self._extract_tier_number(tier_name)
            
            for aspect in tier_aspects:
                aspect_dict = aspect.copy()
                aspect_dict['tier'] = tier_num
                aspect_dict['tier_name'] = tier_name
                aspects.append(aspect_dict)
        
        # Filter by tier if specified
        if tier is not None:
            aspects = [a for a in aspects if a['tier'] == tier]
        
        return aspects
    
    def get_aspect_by_name(self, category_name: str, aspect_name: str) -> Optional[Dict[str, Any]]:
        """
        Get specific aspect configuration by name.
        
        Args:
            category_name: Category name
            aspect_name: Aspect name
        
        Returns:
            Aspect configuration or None if not found
        """
        aspects = self.get_aspects(category_name)
        for aspect in aspects:
            if aspect['name'] == aspect_name:
                return aspect
        return None
    
    def get_brand_config(self, category_name: str) -> Dict[str, Any]:
        """
        Get brand extraction configuration.
        
        Args:
            category_name: Category name
        
        Returns:
            Brand extraction configuration
        """
        config = self.load_category_config(category_name)
        return config.get('brand_extraction', {})
    
    def get_sampling_config(self, category_name: str) -> Dict[str, Any]:
        """
        Get data sampling configuration.
        
        Args:
            category_name: Category name
        
        Returns:
            Sampling configuration
        """
        config = self.load_category_config(category_name)
        return config.get('data', {})
    
    def get_processing_config(self, category_name: str) -> Dict[str, Any]:
        """
        Get processing configuration.
        
        Args:
            category_name: Category name
        
        Returns:
            Processing configuration
        """
        config = self.load_category_config(category_name)
        return config.get('processing', {})
    
    @staticmethod
    def _extract_tier_number(tier_name: str) -> int:
        """Extract tier number from tier name (e.g., 'tier_1_core' -> 1)."""
        if 'tier_1' in tier_name or 'core' in tier_name:
            return 1
        elif 'tier_2' in tier_name or 'common' in tier_name:
            return 2
        elif 'tier_3' in tier_name or 'optional' in tier_name:
            return 3
        return 2  # Default to tier 2
    
    def create_category_from_template(self, category_name: str, amazon_id: str, description: str):
        """
        Create a new category config from template.
        
        Args:
            category_name: Name of new category
            amazon_id: Amazon category ID
            description: Category description
        """
        template_file = self.categories_dir / "_template.yaml"
        new_file = self.categories_dir / f"{category_name.lower()}.yaml"
        
        if new_file.exists():
            raise FileExistsError(f"Category config already exists: {new_file}")
        
        # Read template
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # Replace placeholders
        content = template.replace("{{CATEGORY_NAME}}", category_name)
        content = content.replace("{{AMAZON_CATEGORY_ID}}", amazon_id)
        content = content.replace("{{DESCRIPTION}}", description)
        
        # Write new config
        with open(new_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Created new category config: {new_file}")
        print(f"   Please edit the file to customize aspects and settings.")
    
    def validate_category_config(self, category_name: str) -> bool:
        """
        Validate category configuration.
        
        Args:
            category_name: Category to validate
        
        Returns:
            True if valid, raises exception otherwise
        """
        config = self.load_category_config(category_name)
        
        # Check required fields
        required_fields = ['category', 'data', 'aspects', 'brand_extraction']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Check aspects have at least tier 1
        if 'tier_1_core' not in config['aspects']:
            raise ValueError("Missing tier_1_core aspects")
        
        # Check each aspect has required fields
        for tier_name, aspects in config['aspects'].items():
            for aspect in aspects:
                if 'name' not in aspect:
                    raise ValueError(f"Aspect in {tier_name} missing 'name' field")
                if 'keywords' not in aspect:
                    raise ValueError(f"Aspect '{aspect.get('name')}' missing 'keywords' field")
        
        print(f"✅ Category config '{category_name}' is valid")
        return True


# Example usage
if __name__ == "__main__":
    loader = ConfigLoader()
    
    print("Available categories:", loader.list_available_categories())
    
    # Load electronics config
    electronics_config = loader.load_category_config("electronics")
    print(f"\nElectronics config loaded:")
    print(f"  Category: {electronics_config['category']['name']}")
    print(f"  Top products: {electronics_config['data']['top_products']}")
    
    # Get aspects
    tier1_aspects = loader.get_aspects("electronics", tier=1)
    print(f"\n  Tier 1 aspects: {[a['name'] for a in tier1_aspects]}")
    
    # Validate
    loader.validate_category_config("electronics")
