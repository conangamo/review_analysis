"""Plotly chart components."""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List


def create_aspect_bar_chart(
    aspect_name: str,
    sentiment_data: Dict[str, Any],
    height: int = 100
) -> go.Figure:
    """
    Create horizontal bar chart for aspect sentiment.
    
    Args:
        aspect_name: Name of the aspect
        sentiment_data: Dictionary with sentiment counts and percentages
        height: Chart height
    
    Returns:
        Plotly figure
    """
    positive_pct = sentiment_data.get('positive_pct', 0)
    negative_pct = sentiment_data.get('negative_pct', 0)
    neutral_pct = sentiment_data.get('neutral_pct', 0)
    
    fig = go.Figure(go.Bar(
        x=[positive_pct, negative_pct, neutral_pct],
        y=[aspect_name.replace('_', ' ').title()],
        orientation='h',
        marker=dict(color=['#2ecc71', '#e74c3c', '#95a5a6']),
        text=[f"{positive_pct:.0f}%", f"{negative_pct:.0f}%", f"{neutral_pct:.0f}%"],
        textposition='inside',
        textfont=dict(size=12, color='white'),
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Percentage: %{x:.1f}%<br>'
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        xaxis_title="Percentage",
        showlegend=False,
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(range=[0, 100], showgrid=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_overall_sentiment_pie(
    positive: int,
    negative: int,
    neutral: int
) -> go.Figure:
    """
    Create pie chart for overall sentiment distribution.
    
    Args:
        positive: Positive count
        negative: Negative count
        neutral: Neutral count
    
    Returns:
        Plotly figure
    """
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive, negative, neutral]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(size=14),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        showlegend=True,
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_rating_distribution_chart(rating_dist: Dict[int, int]) -> go.Figure:
    """
    Create bar chart for rating distribution.
    
    Args:
        rating_dist: Dictionary mapping rating (1-5) to count
    
    Returns:
        Plotly figure
    """
    ratings = [1, 2, 3, 4, 5]
    counts = [rating_dist.get(r, 0) for r in ratings]
    
    # Color gradient from red to green
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
    
    fig = go.Figure(go.Bar(
        x=ratings,
        y=counts,
        marker=dict(color=colors),
        text=counts,
        textposition='outside',
        hovertemplate='<b>%{x} Stars</b><br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Rating",
        yaxis_title="Number of Reviews",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        height=300,
        margin=dict(l=50, r=20, t=40, b=50),
        showlegend=False
    )
    
    return fig


def create_aspect_comparison_chart(
    aspects_data: Dict[str, Dict[str, Any]],
    top_n: int = 10
) -> go.Figure:
    """
    Create stacked horizontal bar chart comparing multiple aspects.
    
    Args:
        aspects_data: Dictionary of aspect data
        top_n: Number of top aspects to show
    
    Returns:
        Plotly figure
    """
    # Sort aspects by total mentions
    sorted_aspects = sorted(
        aspects_data.items(),
        key=lambda x: x[1].get('total_mentions', 0),
        reverse=True
    )[:top_n]
    
    aspect_names = [name.replace('_', ' ').title() for name, _ in sorted_aspects]
    positive_pcts = [data.get('positive_pct', 0) for _, data in sorted_aspects]
    negative_pcts = [data.get('negative_pct', 0) for _, data in sorted_aspects]
    neutral_pcts = [data.get('neutral_pct', 0) for _, data in sorted_aspects]
    
    fig = go.Figure()
    
    # Add positive bars
    fig.add_trace(go.Bar(
        name='Positive',
        y=aspect_names,
        x=positive_pcts,
        orientation='h',
        marker=dict(color='#2ecc71'),
        hovertemplate='<b>%{y}</b><br>Positive: %{x:.1f}%<extra></extra>'
    ))
    
    # Add negative bars
    fig.add_trace(go.Bar(
        name='Negative',
        y=aspect_names,
        x=negative_pcts,
        orientation='h',
        marker=dict(color='#e74c3c'),
        hovertemplate='<b>%{y}</b><br>Negative: %{x:.1f}%<extra></extra>'
    ))
    
    # Add neutral bars
    fig.add_trace(go.Bar(
        name='Neutral',
        y=aspect_names,
        x=neutral_pcts,
        orientation='h',
        marker=dict(color='#95a5a6'),
        hovertemplate='<b>%{y}</b><br>Neutral: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='stack',
        xaxis_title="Percentage",
        height=max(400, len(aspect_names) * 40),
        margin=dict(l=150, r=20, t=40, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


# Example usage
if __name__ == "__main__":
    import streamlit as st
    
    # Test data
    test_aspect_data = {
        'positive_pct': 75,
        'negative_pct': 15,
        'neutral_pct': 10,
        'total_mentions': 150
    }
    
    test_aspects = {
        'battery': {'positive_pct': 80, 'negative_pct': 10, 'neutral_pct': 10, 'total_mentions': 200},
        'screen': {'positive_pct': 70, 'negative_pct': 20, 'neutral_pct': 10, 'total_mentions': 180},
        'performance': {'positive_pct': 65, 'negative_pct': 25, 'neutral_pct': 10, 'total_mentions': 150},
        'value': {'positive_pct': 40, 'negative_pct': 50, 'neutral_pct': 10, 'total_mentions': 120},
    }
    
    test_rating_dist = {5: 100, 4: 50, 3: 20, 2: 10, 1: 5}
    
    print("Chart components created successfully!")
    print("\nTo test in Streamlit:")
    print("  streamlit run src/ui/components/charts.py")
