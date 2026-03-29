"""Formatting utilities for UI display."""

from datetime import datetime
from typing import Union


def format_number(num: Union[int, float], decimals: int = 0) -> str:
    """
    Format number with thousands separator.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string (e.g., "1,234" or "1,234.56")
    """
    if num is None:
        return "N/A"
    
    if decimals > 0:
        return f"{num:,.{decimals}f}"
    else:
        return f"{int(num):,}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value between 0-100
        decimals: Number of decimal places
    
    Returns:
        Formatted string (e.g., "75.5%")
    """
    if value is None:
        return "N/A"
    
    return f"{value:.{decimals}f}%"


def format_rating(rating: float) -> str:
    """
    Format rating with star emoji.
    
    Args:
        rating: Rating value
    
    Returns:
        Formatted string (e.g., "4.5 ⭐")
    """
    if rating is None:
        return "N/A"
    
    return f"{rating:.1f} ⭐"


def format_timestamp(timestamp: int) -> str:
    """
    Format unix timestamp to readable date.
    
    Args:
        timestamp: Unix timestamp
    
    Returns:
        Formatted date string (e.g., "Mar 15, 2023")
    """
    if timestamp is None:
        return "N/A"
    
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%b %d, %Y")
    except:
        return "N/A"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated text
    """
    if text is None:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def get_sentiment_color(sentiment: str) -> str:
    """
    Get color for sentiment.
    
    Args:
        sentiment: 'positive', 'negative', or 'neutral'
    
    Returns:
        Color name or hex code
    """
    colors = {
        'positive': '#2ecc71',  # Green
        'negative': '#e74c3c',  # Red
        'neutral': '#95a5a6'    # Gray
    }
    
    return colors.get(sentiment.lower(), '#3498db')  # Blue as default


def get_sentiment_emoji(sentiment: str) -> str:
    """
    Get emoji for sentiment.
    
    Args:
        sentiment: 'positive', 'negative', or 'neutral'
    
    Returns:
        Emoji string
    """
    emojis = {
        'positive': '🟢',
        'negative': '🔴',
        'neutral': '⚪'
    }
    
    return emojis.get(sentiment.lower(), '⚫')


def format_confidence(confidence: float) -> str:
    """
    Format confidence score.
    
    Args:
        confidence: Confidence value (0.0 to 1.0)
    
    Returns:
        Formatted string (e.g., "85%")
    """
    if confidence is None:
        return "N/A"
    
    return f"{confidence * 100:.0f}%"


def format_aspect_name(aspect: str) -> str:
    """
    Format aspect name for display.
    
    Args:
        aspect: Aspect name (e.g., 'battery', 'ease_of_use')
    
    Returns:
        Formatted name (e.g., 'Battery', 'Ease of Use')
    """
    # Replace underscores with spaces and title case
    return aspect.replace('_', ' ').title()


# Example usage
if __name__ == "__main__":
    print("Formatting Utilities Examples:")
    print("="*50)
    
    print(f"\nNumbers:")
    print(f"  1234567 → {format_number(1234567)}")
    print(f"  1234.567 → {format_number(1234.567, 2)}")
    
    print(f"\nPercentages:")
    print(f"  75.5 → {format_percentage(75.5)}")
    print(f"  80.123 → {format_percentage(80.123, 2)}")
    
    print(f"\nRatings:")
    print(f"  4.5 → {format_rating(4.5)}")
    
    print(f"\nTimestamps:")
    print(f"  1678886400 → {format_timestamp(1678886400)}")
    
    print(f"\nText:")
    print(f"  Long text → {truncate_text('This is a very long text that should be truncated', 30)}")
    
    print(f"\nSentiment:")
    print(f"  positive → {get_sentiment_emoji('positive')} {get_sentiment_color('positive')}")
    print(f"  negative → {get_sentiment_emoji('negative')} {get_sentiment_color('negative')}")
    print(f"  neutral → {get_sentiment_emoji('neutral')} {get_sentiment_color('neutral')}")
    
    print(f"\nConfidence:")
    print(f"  0.85 → {format_confidence(0.85)}")
    
    print(f"\nAspect names:")
    print(f"  battery → {format_aspect_name('battery')}")
    print(f"  ease_of_use → {format_aspect_name('ease_of_use')}")
