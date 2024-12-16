import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
import base64

# Risk Analysis
def calculate_risk_metrics(data):
    required_columns = ['revenue', 'expenses', 'cash_flow']
    matches = find_best_match(required_columns, data.columns.tolist())
    
    if None in matches.values():
        st.warning("Missing required columns for Risk Analysis")
        return None
        
    metrics = {}
    for col in matches.values():
        metrics[f'{col}_volatility'] = data[col].std() / data[col].mean()  # Coefficient of variation
        metrics[f'{col}_var_95'] = np.percentile(data[col], 5)  # Value at Risk (95% confidence)
        
    z_scores = stats.zscore(data[matches['revenue']])
    metrics['outliers'] = len([x for x in z_scores if abs(x) > 2])
    
    return pd.DataFrame([metrics])

# Financial Ratios
def calculate_financial_ratios(data):
    required_columns = ['current_assets', 'current_liabilities', 'total_assets', 'total_debt', 'revenue', 'net_income']
    matches = find_best_match(required_columns, data.columns.tolist())
    
    if None in matches.values():
        st.warning("Missing required columns for Financial Ratios")
        return None
        
    ratios = {
        'Current Ratio': data[matches['current_assets']] / data[matches['current_liabilities']],
        'ROA': data[matches['net_income']] / data[matches['total_assets']],
        'Asset Turnover': data[matches['revenue']] / data[matches['total_assets']],
        'Debt Ratio': data[matches['total_debt']] / data[matches['total_assets']]
    }
    
    return pd.DataFrame(ratios)

# Historical Trend Analysis
def analyze_trends(data, column):
    if column not in data.columns:
        st.warning(f"Column {column} not found in dataset")
        return None
        
    # Calculate moving averages
    ma_30 = data[column].rolling(window=30).mean()
    ma_90 = data[column].rolling(window=90).mean()
    
    # Calculate trend
    x = np.arange(len(data))
    slope, intercept, r_value, _, _ = stats.linregress(x, data[column])
    trend_line = slope * x + intercept
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data[column], name='Actual'))
    fig.add_trace(go.Scatter(y=ma_30, name='30-day MA'))
    fig.add_trace(go.Scatter(y=ma_90, name='90-day MA'))
    fig.add_trace(go.Scatter(y=trend_line, name='Trend'))
    
    return fig

# Export Function
def get_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Interactive Visualization
def create_interactive_plot(data, x_col, y_col, color_col=None):
    fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                    trendline="ols", title=f"{y_col} vs {x_col}")
    return fig

# Enhanced Main Function
def main():
    st.title("Enhanced Smart Financial Analysis Tool")
    st.sidebar.header("Configuration")

    uploaded_file = st.sidebar.file_uploader("Upload your financial data (CSV or Excel):")
    if uploaded_file:
        data = load_data(uploaded_file)

        if data is not None:
            st.header("Uploaded Data Preview")
            st.write(data.head())
            
            # Original Analysis Options
            if st.sidebar.checkbox("Show Cash Flow Analysis"):
                st.header("Cash Flow Analysis")
                cash_flow = cash_flow_analysis(data)
                if cash_flow is not None:
                    st.table(cash_flow)
                    st.markdown(get_download_link(cash_flow, "cash_flow_analysis.csv"), unsafe_allow_html=True)
            
            # New Risk Analysis
            if st.sidebar.checkbox("Show Risk Analysis"):
                st.header("Risk Metrics")
                risk_metrics = calculate_risk_metrics(data)
                if risk_metrics is not None:
                    st.table(risk_metrics)
                    fig = px.bar(risk_metrics.melt(), x='variable', y='value')
                    st.plotly_chart(fig)
            
            # Financial Ratios
            if st.sidebar.checkbox("Show Financial Ratios"):
                st.header("Financial Ratios")
                ratios = calculate_financial_ratios(data)
                if ratios is not None:
                    st.table(ratios)
                    fig = px.line(ratios)
                    st.plotly_chart(fig)
            
            # Historical Trends
            if st.sidebar.checkbox("Show Historical Trends"):
                st.header("Historical Trend Analysis")
                column = st.selectbox("Select column for trend analysis", data.columns)
                fig = analyze_trends(data, column)
                if fig is not None:
                    st.plotly_chart(fig)
            
            # Interactive Visualization
            if st.sidebar.checkbox("Create Custom Visualization"):
                st.header("Custom Visualization")
                x_col = st.selectbox("Select X-axis", data.columns)
                y_col = st.selectbox("Select Y-axis", data.columns)
                color_col = st.selectbox("Select Color Variable (optional)", ['None'] + list(data.columns))
                
                color_col = None if color_col == 'None' else color_col
                fig = create_interactive_plot(data, x_col, y_col, color_col)
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()