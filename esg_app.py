import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
from datetime import datetime, timedelta
import random
from anthropic import Anthropic
import logging

# Import the functions from your original code
# Assuming these functions are in a file named 'stock_news_functions.py'
from stock_news_functions import process_company_news, create_stock_chart_with_news

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app
def main():
    st.title("Company Stock and News Analysis")

    company_name = st.text_input("Enter company name:", "Tesla")
    news_api_key = st.sidebar.text_input("News API Key:", "b8477effcf6f4bd2a1b7d8aa05ba8bae", type="password")
    claude_api_key = st.sidebar.text_input("Claude API Key:", "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", type="password")
    alpha_vantage_api_key = st.sidebar.text_input("Alpha Vantage API Key:", "4ZVEUCTDG9JRYSNV", type="password")

    if st.button("Analyze"):
        with st.spinner("Processing company news and stock data..."):
            status_placeholder = st.empty()
            
            try:
                status_placeholder.text("1. Retrieving headlines...")
                result = process_company_news(company_name, claude_api_key, news_api_key, status_placeholder)
                
                if result is None:
                    st.error("process_company_news returned None. Check the logs for more information.")
                else:
                    summary, deduplicated_articles = result
                    
                    status_placeholder.text("5. Producing final summary...not long now!")
                    st.subheader("News Summary")
                    st.write(summary)

                    if deduplicated_articles:
                        status_placeholder.text("Creating stock chart...")
                        chart = create_stock_chart_with_news(company_name, alpha_vantage_api_key, deduplicated_articles)
                        if chart:
                            st.subheader("Stock Price with News Events")
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.error(f"Failed to create chart for {company_name}")
                    else:
                        st.warning(f"No articles found for {company_name}. Unable to create chart.")

                status_placeholder.empty()  # Clear the status message when done

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Traceback:")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()