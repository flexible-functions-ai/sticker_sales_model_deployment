# Import required libraries
import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
from datetime import datetime
import io

def load_and_predict_data(csv_path):
    """
    Sends the test CSV to the API endpoint and gets predictions
    """
    try:
        # Read the CSV file
        test_df = pd.read_csv(csv_path)
        
        # Set the API URL
        api_url = "https://flexible-functions-ai--sticker-sales-api-predict-csv.modal.run"
        
        # Prepare the file for upload using proper multipart/form-data format
        with open(csv_path, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            
            # Make the request
            st.info(f"Sending data to API at {api_url}...")
            response = requests.post(api_url, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            try:
                result = response.json()
                
                # Check if the result is an error message
                if isinstance(result, dict) and not result.get('success', True):
                    st.error(f"Error from API: {result.get('error', 'Unknown error')}")
                    st.error("Using dummy predictions for demonstration")
                    test_df['predicted_sales'] = 100  # Dummy predictions
                else:
                    # Assume the result is a list of predictions
                    if isinstance(result, list) and len(result) == len(test_df):
                        st.success("Successfully received predictions from API")
                        test_df['predicted_sales'] = result
                    else:
                        st.warning(f"Unexpected response format. Using dummy predictions.")
                        st.json(result)  # Show the actual response
                        test_df['predicted_sales'] = 100  # Dummy predictions
            except Exception as e:
                st.error(f"Error parsing API response: {str(e)}")
                st.text(f"Response text: {response.text[:500]}")  # Show first 500 chars
                test_df['predicted_sales'] = 100  # Dummy predictions
        else:
            st.error(f"API returned status code: {response.status_code}")
            st.text(f"Response text: {response.text[:500]}")  # Show first 500 chars
            test_df['predicted_sales'] = 100  # Dummy predictions
        
        # Convert date column to datetime
        test_df['date'] = pd.to_datetime(test_df['date'])
        
        return test_df
    
    except Exception as e:
        st.error(f"Error processing prediction request: {str(e)}")
        # Return a dummy dataframe for demonstration
        test_df = pd.read_csv(csv_path)
        test_df['predicted_sales'] = 100  # Dummy prediction values
        test_df['date'] = pd.to_datetime(test_df['date'])
        return test_df

# Keep the rest of your dashboard.py file the same


def create_dashboard():
    """
    Creates the Streamlit dashboard with enhanced filters, KPI cards, and visualizations
    """
    st.title("Sales Prediction Dashboard")
    
    # Add custom CSS for dark theme cards
    st.markdown("""
        <style>
        .metric-card {
            background-color: #2C3333;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .metric-label {
            color: #718096;
            font-size: 0.875rem;
        }
        .metric-value {
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .trend-positive {
            color: #48BB78;
        }
        .trend-negative {
            color: #F56565;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # File uploader for the test CSV
    uploaded_file = st.file_uploader("Upload test CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open('temp_test.csv', 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Load data and get predictions
        df = load_and_predict_data('temp_test.csv')
        
        # Convert date column to datetime if not already
        df['date'] = pd.to_datetime(df['date'])
        
        # Creating filters in a sidebar
        st.sidebar.header("Filters")
        
        # Time period filter
        time_periods = {
            'All Time': None,
            'Last Month': 30,
            'Last 3 Months': 90,
            'Last Year': 365
        }
        selected_period = st.sidebar.selectbox('Select Time Period', list(time_periods.keys()))
        
        # Country filter
        countries = ['All'] + sorted(df['country'].unique().tolist())
        selected_country = st.sidebar.selectbox('Select Country', countries)
        
        # Store filter
        stores = ['All'] + sorted(df['store'].unique().tolist())
        selected_store = st.sidebar.selectbox('Select Store', stores)
        
        # Product filter
        products = ['All'] + sorted(df['product'].unique().tolist())
        selected_product = st.sidebar.selectbox('Select Product', products)
        
        # Apply filters
        filtered_df = df.copy()
        
        # Apply time filter
        if time_periods[selected_period]:
            max_date = filtered_df['date'].max()
            cutoff_date = max_date - pd.Timedelta(days=time_periods[selected_period])
            filtered_df = filtered_df[filtered_df['date'] >= cutoff_date]
        
        if selected_country != 'All':
            filtered_df = filtered_df[filtered_df['country'] == selected_country]
        if selected_store != 'All':
            filtered_df = filtered_df[filtered_df['store'] == selected_store]
        if selected_product != 'All':
            filtered_df = filtered_df[filtered_df['product'] == selected_product]
        
        # Calculate metrics for KPI cards
        total_sales = filtered_df['predicted_sales'].sum()
        avg_daily_sales = filtered_df.groupby('date')['predicted_sales'].sum().mean()
        
        # Calculate period-over-period changes
        if time_periods[selected_period]:
            previous_period = filtered_df['date'].min() - pd.Timedelta(days=time_periods[selected_period])
            previous_df = df[df['date'] >= previous_period]
            previous_df = previous_df[previous_df['date'] < filtered_df['date'].min()]
            
            prev_total_sales = previous_df['predicted_sales'].sum()
            sales_change = ((total_sales - prev_total_sales) / prev_total_sales * 100 
                          if prev_total_sales != 0 else 0)
        else:
            sales_change = 0
            
        # Create KPI cards using columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Predicted Sales</div>
                    <div class="metric-value">${total_sales:,.0f}</div>
                    <div class="{'trend-positive' if sales_change >= 0 else 'trend-negative'}">
                        {sales_change:+.1f}% vs previous period
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Daily Sales</div>
                    <div class="metric-value">${avg_daily_sales:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            top_store = (filtered_df.groupby('store')['predicted_sales']
                        .sum().sort_values(ascending=False).index[0])
            store_sales = (filtered_df.groupby('store')['predicted_sales']
                         .sum().sort_values(ascending=False).iloc[0])
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Top Performing Store</div>
                    <div class="metric-value">{top_store}</div>
                    <div class="metric-label">${store_sales:,.0f} in sales</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            top_product = (filtered_df.groupby('product')['predicted_sales']
                          .sum().sort_values(ascending=False).index[0])
            product_sales = (filtered_df.groupby('product')['predicted_sales']
                           .sum().sort_values(ascending=False).iloc[0])
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Best Selling Product</div>
                    <div class="metric-value">{top_product}</div>
                    <div class="metric-label">${product_sales:,.0f} in sales</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Group by date and calculate daily total predicted sales
        daily_sales = filtered_df.groupby('date')['predicted_sales'].sum().reset_index()
        
        # Create the line chart using Plotly with dark theme
        fig = px.line(
            daily_sales,
            x='date',
            y='predicted_sales',
            title='Predicted Daily Sales Over Time'
        )
        
        # Update layout for dark theme
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date",
            yaxis_title="Predicted Sales",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add trend line
        fig.add_scatter(
            x=daily_sales['date'],
            y=daily_sales['predicted_sales'].rolling(7).mean(),
            name='7-day trend',
            line=dict(dash='dash', color='#48BB78'),
            visible='legendonly'
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed data view
        st.subheader("Detailed Data View")
        st.dataframe(
            filtered_df.sort_values('date'),
            hide_index=True
        )

if __name__ == "__main__":
    # Set page configuration at the very beginning
    st.set_page_config(
        page_title="Sales Prediction Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    create_dashboard()