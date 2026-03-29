"""Aspect management utilities."""

from typing import List, Dict, Any, Optional

try:
    from .config_loader import ConfigLoader
except ImportError:
    # For direct execution
    from config_loader import ConfigLoader


class AspectManager:
    """Manage aspects for a category."""
    
    def __init__(self, category_name: str, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize aspect manager.
        
        Args:
            category_name: Category name
            config_loader: Optional config loader instance
        """
        if config_loader is None:
            config_loader = ConfigLoader()
        
        self.category_name = category_name
        self.config_loader = config_loader
        self.aspects = config_loader.get_aspects(category_name)
        
        # Build keyword index for fast lookup
        self.keyword_index = self._build_keyword_index()
    
    def _build_keyword_index(self) -> Dict[str, List[str]]:
        """
        Build keyword to aspect name index.
        
        Returns:
            Dictionary mapping keywords to aspect names
        """
        index = {}
        
        for aspect in self.aspects:
            aspect_name = aspect['name']
            for keyword in aspect.get('keywords', []):
                keyword_lower = keyword.lower()
                if keyword_lower not in index:
                    index[keyword_lower] = []
                index[keyword_lower].append(aspect_name)
        
        return index
    
    def get_aspects_by_tier(self, tier: int) -> List[Dict[str, Any]]:
        """
        Get all aspects for a specific tier.
        
        Args:
            tier: Tier number (1, 2, or 3)
        
        Returns:
            List of aspect configurations
        """
        return [a for a in self.aspects if a['tier'] == tier]
    
    def get_aspect(self, aspect_name: str) -> Optional[Dict[str, Any]]:
        """
        Get aspect configuration by name.
        
        Args:
            aspect_name: Aspect name
        
        Returns:
            Aspect configuration or None
        """
        for aspect in self.aspects:
            if aspect['name'] == aspect_name:
                return aspect
        return None
    
    def detect_aspects_by_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect aspects in text using keyword matching.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of detected aspects
        """
        text_lower = text.lower()
        detected = set()
        
        # Check each aspect's keywords
        for aspect in self.aspects:
            for keyword in aspect.get('keywords', []):
                if keyword.lower() in text_lower:
                    detected.add(aspect['name'])
                    break
        
        # Return aspect configs for detected aspects
        return [a for a in self.aspects if a['name'] in detected]
    
    def get_aspects_for_analysis(
        self, 
        text: str, 
        include_tier1_always: bool = False,  # 🔧 FIX: Changed from True to False
        strict_keyword_matching: bool = True  # 🔧 NEW: Require keyword match for all tiers
    ) -> List[Dict[str, Any]]:
        """
        Get aspects to analyze for given text.
        
        Strategy (IMPROVED):
        - All tiers: Only include if keywords detected (strict mode)
        - OR include tier 1 if strong overall sentiment signal detected
        
        Args:
            text: Review text
            include_tier1_always: DEPRECATED - Always include tier 1 aspects (now defaults to False)
            strict_keyword_matching: If True, require keyword match for ALL aspects
        
        Returns:
            List of aspects to analyze
        """
        aspects_to_analyze = []
        
        # Detect aspects by keywords (all tiers)
        detected = self.detect_aspects_by_keywords(text)
        
        if strict_keyword_matching:
            # STRICT MODE: Only analyze aspects with keyword matches
            aspects_to_analyze = detected
        else:
            # LENIENT MODE: Include detected + tier 1 if strong signal
            aspects_to_analyze = detected.copy()
            
            # Add tier 1 aspects if explicitly requested OR strong sentiment detected
            if include_tier1_always or self._has_strong_sentiment_signal(text):
                tier1_aspects = self.get_aspects_by_tier(1)
                for aspect in tier1_aspects:
                    if aspect not in aspects_to_analyze:
                        aspects_to_analyze.append(aspect)
        
        return aspects_to_analyze
    
    def _has_strong_sentiment_signal(self, text: str) -> bool:
        """
        Detect if review has strong overall sentiment signal.
        Useful for tier 1 aspects that might be implied without explicit keywords.
        
        Args:
            text: Review text
        
        Returns:
            True if strong sentiment words detected
        """
        text_lower = text.lower()
        
        # Strong positive indicators
        positive_words = [
            'excellent', 'amazing', 'perfect', 'outstanding', 'fantastic',
            'love', 'awesome', 'incredible', 'superb', 'wonderful'
        ]
        
        # Strong negative indicators
        negative_words = [
            'terrible', 'awful', 'horrible', 'worst', 'useless',
            'broken', 'defective', 'junk', 'garbage', 'disappointed'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Require at least 2 strong sentiment words to consider as "strong signal"
        return (pos_count + neg_count) >= 2
    
    def should_display_aspect(
        self, 
        aspect_name: str, 
        mention_count: int
    ) -> bool:
        """
        Determine if aspect should be displayed based on mention count.
        
        Args:
            aspect_name: Aspect name
            mention_count: Number of times aspect was mentioned
        
        Returns:
            True if should display
        """
        aspect = self.get_aspect(aspect_name)
        if not aspect:
            return False
        
        # Tier 1: Always display
        if aspect['tier'] == 1:
            return True
        
        # Tier 2/3: Check minimum mentions
        min_mentions = aspect.get('min_mentions', 5)
        return mention_count >= min_mentions
    
    def get_aspect_priority(self, aspect_name: str) -> int:
        """
        Get display priority for an aspect.
        
        Args:
            aspect_name: Aspect name
        
        Returns:
            Priority (lower is higher priority)
        """
        aspect = self.get_aspect(aspect_name)
        if not aspect:
            return 999
        
        return aspect.get('priority', aspect['tier'] * 100)
    
    def sort_aspects_by_priority(
        self, 
        aspects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sort aspects by display priority.
        
        Args:
            aspects: List of aspects
        
        Returns:
            Sorted list
        """
        return sorted(
            aspects, 
            key=lambda a: (a['tier'], a.get('priority', 999))
        )
    
    def get_all_aspect_names(self) -> List[str]:
        """Get list of all aspect names."""
        return [a['name'] for a in self.aspects]
    
    def get_aspect_keywords(self, aspect_name: str) -> List[str]:
        """Get keywords for a specific aspect."""
        aspect = self.get_aspect(aspect_name)
        return aspect.get('keywords', []) if aspect else []
    
    def print_aspect_summary(self):
        """Print summary of all aspects."""
        print(f"\n📋 Aspects for {self.category_name}")
        print("=" * 60)
        
        for tier in [1, 2, 3]:
            tier_aspects = self.get_aspects_by_tier(tier)
            if not tier_aspects:
                continue
            
            tier_name = tier_aspects[0].get('tier_name', f'tier_{tier}')
            print(f"\n{tier_name.upper().replace('_', ' ')}:")
            
            for aspect in tier_aspects:
                keywords = ', '.join(aspect['keywords'][:3])
                if len(aspect['keywords']) > 3:
                    keywords += ', ...'
                
                priority = aspect.get('priority', 'N/A')
                min_mentions = aspect.get('min_mentions', 'N/A')
                
                # Format priority and min_mentions properly (handle both int and str)
                priority_str = str(priority) if priority != 'N/A' else 'N/A'
                min_mentions_str = str(min_mentions) if min_mentions != 'N/A' else 'N/A'
                
                print(f"  • {aspect['name']:15s} | "
                      f"Priority: {priority_str:>3s} | "
                      f"Min mentions: {min_mentions_str:>3s} | "
                      f"Keywords: {keywords}")
        
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Test aspect manager
    manager = AspectManager("electronics")
    
    # Print summary
    manager.print_aspect_summary()
    
    # Test aspect detection
    test_text = "The battery life is amazing! Screen is bright and clear, but price is too high."
    
    print(f"\n\nTest Text: {test_text}")
    print("\nDetected aspects:")
    
    detected = manager.detect_aspects_by_keywords(test_text)
    for aspect in detected:
        print(f"  • {aspect['name']} (Tier {aspect['tier']})")
    
    print("\nAspects to analyze:")
    to_analyze = manager.get_aspects_for_analysis(test_text)
    for aspect in to_analyze:
        print(f"  • {aspect['name']} (Tier {aspect['tier']})")
