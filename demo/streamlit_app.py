"""
Streamlit demo application for Explainable AutoML.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from typing import Dict, Any, Optional

# Set page config
st.set_page_config(
    page_title="Explainable AutoML Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_results() -> Optional[Dict[str, Any]]:
    """Load results from saved file."""
    try:
        if os.path.exists("assets/results.pkl"):
            return joblib.load("assets/results.pkl")
        return None
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return None

def create_feature_importance_plot(explanations: Dict[str, Any]) -> go.Figure:
    """Create feature importance plot."""
    if "shap_values_summary" in explanations:
        importance_scores = np.abs(explanations["shap_values_summary"]).mean(axis=0)
    elif "importance_mean" in explanations:
        importance_scores = explanations["importance_mean"]
    else:
        return None
    
    feature_names = explanations.get("feature_names", [f"Feature_{i}" for i in range(len(importance_scores))])
    
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1]
    sorted_importance = importance_scores[indices]
    sorted_features = [feature_names[i] for i in indices]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_features,
            y=sorted_importance,
            marker_color='lightblue',
            text=[f"{val:.3f}" for val in sorted_importance],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig

def create_performance_metrics_plot(metrics: Dict[str, float]) -> go.Figure:
    """Create performance metrics plot."""
    # Select key metrics
    key_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    available_metrics = {k: v for k, v in metrics.items() if k in key_metrics}
    
    if not available_metrics:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(available_metrics.keys()),
            y=list(available_metrics.values()),
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(available_metrics)],
            text=[f"{val:.3f}" for val in available_metrics.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_explanation_quality_plot(xai_metrics: Dict[str, float]) -> go.Figure:
    """Create explanation quality plot."""
    # Select XAI metrics
    xai_key_metrics = ["deletion_auc", "insertion_auc", "stability_mean_correlation"]
    available_xai_metrics = {k: v for k, v in xai_metrics.items() if k in xai_key_metrics}
    
    if not available_xai_metrics:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(available_xai_metrics.keys()),
            y=list(available_xai_metrics.values()),
            marker_color='lightgreen',
            text=[f"{val:.3f}" for val in available_xai_metrics.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Explanation Quality Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score",
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Explainable AutoML Demo</h1>', unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Safety Disclaimer</h4>
        <p><strong>This is a research/educational tool. Not for production decisions or control.</strong></p>
        <p>Ensure proper consent and privacy protection when using with real data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Model Performance", "Explanations", "Interactive Analysis"]
    )
    
    # Load results
    results = load_results()
    
    if results is None:
        st.error("No results found. Please run the pipeline first using: `python scripts/run_pipeline.py`")
        st.stop()
    
    # Extract data
    model_metrics = results.get("model_metrics", {})
    xai_metrics = results.get("xai_metrics", {})
    explanations = results.get("explanations", {})
    config = results.get("config", {})
    
    if page == "Overview":
        st.header("📊 Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            if model_metrics:
                fig_perf = create_performance_metrics_plot(model_metrics)
                if fig_perf:
                    st.plotly_chart(fig_perf, use_container_width=True)
                else:
                    st.info("No performance metrics available")
            else:
                st.info("No model metrics available")
        
        with col2:
            st.subheader("Explanation Quality")
            if xai_metrics:
                fig_xai = create_explanation_quality_plot(xai_metrics)
                if fig_xai:
                    st.plotly_chart(fig_xai, use_container_width=True)
                else:
                    st.info("No XAI metrics available")
            else:
                st.info("No explanation quality metrics available")
        
        # Configuration info
        st.subheader("Configuration")
        if config:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dataset", config.get("data", {}).get("name", "Unknown"))
                st.metric("Model", config.get("model", {}).get("_target_", "Unknown").split('.')[-1])
            with col2:
                st.metric("Explainer", config.get("explainer", {}).get("_target_", "Unknown").split('.')[-1])
                st.metric("Seed", config.get("experiment", {}).get("seed", "Unknown"))
            with col3:
                st.metric("Test Size", f"{config.get('data', {}).get('test_size', 0.2)*100:.0f}%")
                st.metric("Generations", config.get("model", {}).get("generations", "Unknown"))
    
    elif page == "Model Performance":
        st.header("🎯 Model Performance")
        
        if model_metrics:
            # Detailed metrics table
            st.subheader("Detailed Metrics")
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame([
                {"Metric": "Accuracy", "Value": model_metrics.get("accuracy", 0)},
                {"Metric": "Precision (Macro)", "Value": model_metrics.get("precision_macro", 0)},
                {"Metric": "Recall (Macro)", "Value": model_metrics.get("recall_macro", 0)},
                {"Metric": "F1-Score (Macro)", "Value": model_metrics.get("f1_macro", 0)},
                {"Metric": "ROC-AUC (OVR)", "Value": model_metrics.get("roc_auc_ovr", 0)},
                {"Metric": "Log Loss", "Value": model_metrics.get("log_loss", 0)},
            ])
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Performance plot
            fig_perf = create_performance_metrics_plot(model_metrics)
            if fig_perf:
                st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("No model performance metrics available")
    
    elif page == "Explanations":
        st.header("🔍 Model Explanations")
        
        if explanations:
            # Feature importance
            st.subheader("Feature Importance")
            fig_importance = create_feature_importance_plot(explanations)
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("No feature importance data available")
            
            # Explanation method info
            st.subheader("Explanation Method")
            method = explanations.get("method", "Unknown")
            st.info(f"Using {method} for explanations")
            
            # XAI metrics
            if xai_metrics:
                st.subheader("Explanation Quality")
                fig_xai = create_explanation_quality_plot(xai_metrics)
                if fig_xai:
                    st.plotly_chart(fig_xai, use_container_width=True)
                
                # Detailed XAI metrics
                st.subheader("Detailed XAI Metrics")
                xai_df = pd.DataFrame([
                    {"Metric": "Faithfulness (Deletion AUC)", "Value": xai_metrics.get("deletion_auc", 0)},
                    {"Metric": "Faithfulness (Insertion AUC)", "Value": xai_metrics.get("insertion_auc", 0)},
                    {"Metric": "Stability (Mean Correlation)", "Value": xai_metrics.get("stability_mean_correlation", 0)},
                    {"Metric": "Stability (Std Correlation)", "Value": xai_metrics.get("stability_std_correlation", 0)},
                ])
                st.dataframe(xai_df, use_container_width=True)
        else:
            st.info("No explanations available")
    
    elif page == "Interactive Analysis":
        st.header("🔬 Interactive Analysis")
        
        st.subheader("Feature Analysis")
        
        if explanations and "feature_names" in explanations:
            feature_names = explanations["feature_names"]
            
            # Feature selection
            selected_feature = st.selectbox(
                "Select a feature to analyze:",
                feature_names
            )
            
            if selected_feature:
                st.info(f"Analyzing feature: {selected_feature}")
                
                # Get feature importance
                if "shap_values_summary" in explanations:
                    importance_scores = np.abs(explanations["shap_values_summary"]).mean(axis=0)
                    feature_idx = feature_names.index(selected_feature)
                    feature_importance = importance_scores[feature_idx]
                    
                    st.metric("Feature Importance", f"{feature_importance:.4f}")
                    
                    # Feature distribution (if we had the data)
                    st.info("Feature distribution analysis would go here with actual data")
        
        # Model comparison
        st.subheader("Model Comparison")
        st.info("Compare different models and their explanations here")
        
        # Add comparison options
        compare_models = st.checkbox("Enable model comparison")
        if compare_models:
            st.info("Model comparison functionality would be implemented here")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Author: <strong>kryptologyst</strong> | GitHub: <a href="https://github.com/kryptologyst">https://github.com/kryptologyst</a></p>
        <p><em>Explainable AutoML Framework for Research and Education</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
