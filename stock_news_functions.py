import requests
from datetime import datetime, timedelta
import logging
from anthropic import Anthropic
from bs4 import BeautifulSoup
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from fake_useragent import UserAgent
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random



def process_company_news(company_name, claude_api_key, news_api_key, status_placeholder):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize Claude client
    anthropic = Anthropic(api_key=claude_api_key)

    # Get the date one month ago in YYYY-MM-DD format
    date_one_week_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Construct the news API URL
    final_url = f'https://newsapi.org/v2/everything?q={company_name}&from={date_one_week_ago}&language=en&apiKey={news_api_key}'

    def extract_date(article):
        date_string = article.get('publishedAt')
        if date_string:
            try:
                # Parse the ISO 8601 formatted date string
                date = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
                # Format the date as desired, e.g., 'YYYY-MM-DD'
                return date.strftime('%Y-%m-%d')
            except ValueError:
                logger.error(f"Failed to parse date: {date_string}")
        return None

    def is_esg_related(headline):
            prompt = f"""Analyze the given headline and determine if it is directly related to Environmental, Social, or Governance (ESG) issues. ESG encompasses specific factors, including but not limited to:
        
        Environmental:
        - Climate change initiatives or carbon emissions reduction
        - Renewable energy adoption or energy efficiency improvements
        - Pollution prevention and control measures
        - Waste management and recycling programs
        - Water conservation and management initiatives
        - Biodiversity protection efforts
        - Sustainable resource use practices
        
        Social:
        - Labor practices and working conditions (e.g., fair wages, safe working environments)
        - Employee health, safety, and wellbeing programs
        - Diversity, equity, and inclusion initiatives
        - Human rights policies and practices
        - Community engagement and social impact programs
        - Data privacy and security measures
        - Product safety and quality improvements
        
        Governance:
        - Board structure, diversity, and independence
        - Executive compensation policies
        - Shareholder rights and engagement practices
        - Business ethics and anti-corruption measures
        - Transparency in reporting and disclosure practices
        - Risk management systems
        - Regulatory compliance initiatives
        - Controversy related news
        - Antitrust related news
        - Sanctioning related news
        
        Instructions:
        1. The headline must explicitly mention or directly imply one of these ESG factors.
        2. Do not infer ESG relevance from general business activities or innovations unless they clearly relate to ESG goals.
        3. Headlines about general business operations, product launches, or financial performance are not ESG-related unless they explicitly mention ESG impacts.
        4. If you deem it is ESG related, begin your response with 'yes' and briefly explain which specific ESG factor it relates to.
        5. If you do not deem it is ESG related, respond with 'no' and nothing else.
        
        Headline: '{headline}'
        """
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = anthropic.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=300,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    if response.content and len(response.content) > 0:
                        model_response = response.content[0].text.strip().lower()
                        logger.info(f"Headline: {headline}")
                        logger.info(f"Model's Response: {model_response}")
                        if "yes" in model_response:
                            return "Add"
                    return "Do not add"
                except Exception as e:
                    logger.error(f"Error processing headline '{headline}': {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"Failed to process headline after {max_retries} attempts")
                        return "Do not add"












    def can_parse_url(url):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                BeautifulSoup(response.text, 'html.parser')
                return True
        except Exception:
            pass
        return False

    def are_headlines_similar(headline1, headline2):
        prompt = f"""Compare the following two headlines and determine if they are discussing the same news topic or event. Respond with only 'yes' if they are similar, or 'no' if they are not. Do not provide any explanation.
        Headline 1: {headline1}
        Headline 2: {headline2}
        Are these headlines discussing the same topic or event?"""
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text.strip().lower() == 'yes'



    def deduplicate_articles(filtered_articles):
        deduplicated_articles = []
        for article in filtered_articles:
            is_duplicate = False
            for existing_article in deduplicated_articles:
                if are_headlines_similar(article['title'], existing_article['title']):
                    is_duplicate = True
                    if can_parse_url(article['url']) and not can_parse_url(existing_article['url']):
                        # Replace existing article with the new one if its URL is parseable
                        deduplicated_articles.remove(existing_article)
                        deduplicated_articles.append(article)
                    break
            if not is_duplicate:
                deduplicated_articles.append(article)
        
        return deduplicated_articles


    def fetch_and_scrape_articles(filtered_articles):
        my_articles = []
        total_articles = len(filtered_articles)
        ua = UserAgent()
        
        # Set up a session with retry strategy
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        for index, article in enumerate(filtered_articles, start=1):
            url = article['url']
            title = article['title']
            published_date = article.get('date', 'Unknown date')
            
            # Check if the article is from www.investing.com
            if 'www.investing.com' in url:
                logger.info(f"Skipping Article {index} ({title}) from www.investing.com")
                continue
            
            logger.info(f"Fetching content for Article {index} of {total_articles}: {title}")
            
            headers = {
                'User-Agent': ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
            }
            try:
                response = session.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                # Try multiple methods to find the article content
                content = None
                for selector in ['article', '.article-content', '#article-body', '.story-body']:
                    content = soup.select_one(selector)
                    if content:
                        break
                if not content:
                    # If no specific article container found, try to get the main content
                    content = soup.find('main') or soup.find('body')
                if content:
                    # Remove unwanted elements
                    for unwanted in content.select('script, style, nav, header, footer, [class*="ad"], .comments'):
                        unwanted.decompose()
                    article_text = ' '.join(content.stripped_strings)
                    if article_text:
                        my_articles.append({
                            'title': title,
                            'content': article_text,
                            'published_date': published_date,
                            'url': url
                        })
                        logger.info(f"Fetching complete for Article {index} ({title})")
                    else:
                        logger.warning(f"Article {index} ({title}) was scraped but content is empty.")
                else:
                    logger.warning(f"Content not found for Article {index} ({title}).")
                time.sleep(2)  # Be polite, add a delay between requests
            except requests.HTTPError as http_err:
                logger.error(f"HTTP error occurred for Article {index} ({title}): {http_err}")
            except requests.Timeout:
                logger.error(f"Timeout occurred while fetching Article {index} ({title})")
            except Exception as err:
                logger.error(f"Error occurred while processing Article {index} ({title}): {str(err)}")
        return my_articles
    
    def summarize_article(article_content):
        # Use GPT-4 to generate a summary
        prompt = f"""Your task is to summarise a news article in no more than 300 words,
        making full use of this word limit. Ignore boilerplate text and try to really focus on the message.
        You should use as much detail as possible to make use of the 300 word limit.
        Don't be afraid to output 300 words!Summarise the following: {article_content}
        """
        
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=450,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        model_response = response.content[0].text.strip().lower()
        
        return model_response


    def generate_financial_summary(summarized_articles):
        client = Anthropic(api_key=claude_api_key)
        
        # Join summaries with titles and dates
        joined_summaries = "\n\n".join([
            f"Title: {article['title']}\n"
            f"Date: {article['published_date']}\n"
            f"Summary: {article['summary']}"
            for article in summarized_articles
        ])
        
        prompt = f"""Please provide a comprehensive and detailed summary of the following points extracted from this week's ESG news. Your summary should be in the style of an in-depth financial news report, covering all major topics and themes present in the provided information.
    Key requirements:
    1. Start directly with the content. Do not include any introductory phrases like "Here is a summary" or mention the word count. This is very important! Start directly with the content.
    2. Do not begin your response with 'Here is'.
    3. Begin your response directly with the summary.
    4. Consolidate and elaborate on related topics: If you encounter the same subject mentioned in different parts of the context, discuss them together, providing a thorough analysis of the topic's various aspects and implications.
    5. Your summary should aim to retain 70% of the original content. 
    6. Where the news mentions any data or figures, include them in the summary. Examples include details of product recalls or the value of financial penalties. We want to retain as much quantitative information as possible.
    7. Structure your report: Organize the information into clear sections or themes.
    9. Where there are multiple topics, place a header before each topic.
    10. Add a touch of humor: Towards the end of your summary, include a light-hearted observation or witty remark related to one of the news items, but ensure it doesn't detract from the overall professional tone of the report. Do not place a header above this last humour section.
    Your summary should be informative, engaging, and provide valuable insights for readers seeking a comprehensive understanding of the week's financial news. Aim for a length that thoroughly covers all significant points while maintaining reader interest.
    Here are the points to summarize:{joined_summaries}"""
    
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=3000,  # Increased to allow for a more detailed response
            system="You are a brilliant, Pulitzer-winning financial journalist who writes in-depth, analytical pieces.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        summary = response.content[0].text
        
        return summary

    # Main process
    response = requests.get(final_url)
    articles = response.json()['articles'] if response.status_code == 200 else []
    articles = [article for article in articles if company_name.lower() in article['title'].lower() or f"{company_name}'s".lower() in article['title'].lower()]

    status_placeholder.text("2. Identifying ESG headlines...")
    esg_headlines = []
    for article in articles:
        headline = article['title']
        date = extract_date(article)
        outcome = is_esg_related(headline)
        if outcome == 'Add':
            logger.info(f"Adding to esg_headlines: {headline} (Date: {date})")
            esg_headlines.append((headline, date))
    
    status_placeholder.text(f"2. {len(esg_headlines)} ESG headlines identified")
    
    filtered_articles = [
        {
            'title': article['title'],
            'date': extract_date(article),
            'content': article.get('content', ''),
            'url': article.get('url', '')
        }
        for article in articles
        if article['title'] in [h[0] for h in esg_headlines]
    ]


    if not filtered_articles:
        logger.info(f"No relevant ESG headlines found for {company_name}")
        return f"No relevant ESG news found for {company_name} in the past week.", []

    status_placeholder.text("3. De-duplicating headlines...")
    deduplicated_filtered_articles = deduplicate_articles(filtered_articles)
    deduplicated_filtered_articles.sort(key=lambda x: x.get('date') or '', reverse=True)

    status_placeholder.text("4. Scraping articles...")
    summy = fetch_and_scrape_articles(deduplicated_filtered_articles)
    if not summy:
        logger.warning(f"No articles could be successfully scraped for {company_name}")
        return f"No articles could be successfully scraped for {company_name}. Unable to generate summary.", []

    status_placeholder.text("5. Summarising articles...")
    summarized_articles = []
    for article in summy:
        summary = summarize_article(article['content'])
        summarized_articles.append({
            'title': article['title'],
            'published_date': article['published_date'],
            'url': article['url'],
            'summary': summary
        })

    status_placeholder.text("5. Producing final summary...not long now!")

    overall_summary = generate_financial_summary(summarized_articles)    
    
    return overall_summary, deduplicated_filtered_articles


#Now we make an additional function to create the charts and wrappers for doing it in one package

def search_ticker(keyword, api_key):
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={keyword}&apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        best_matches = data.get('bestMatches', [])
        
        if not best_matches:
            print(f"No ticker found for {keyword}")
            return None

        # Function to safely get a value from a dictionary
        def safe_get(dict_obj, key, default=''):
            return dict_obj.get(key, default)

        # Prioritize NYSE and NASDAQ listings
        us_exchanges = ['NYSE', 'NASDAQ']
        us_listings = [
            match for match in best_matches 
            if safe_get(match, '4. region') == 'United States' and safe_get(match, '3. type') == 'Equity'
        ]
        
        if us_listings:
            # Sort by exchange priority and then by highest match score
            sorted_listings = sorted(
                us_listings, 
                key=lambda x: (
                    us_exchanges.index(safe_get(x, '8. exchange')) if safe_get(x, '8. exchange') in us_exchanges else len(us_exchanges),
                    -float(safe_get(x, '9. matchScore', '0'))
                )
            )
            best_match = sorted_listings[0]
        else:
            # If no US listings, just take the highest match score
            best_match = max(best_matches, key=lambda x: float(safe_get(x, '9. matchScore', '0')))

        ticker = safe_get(best_match, '1. symbol')
        exchange = safe_get(best_match, '8. exchange', 'Unknown Exchange')
        print(f"Selected ticker for {keyword}: {ticker} ({exchange})")
        return ticker

    except requests.RequestException as e:
        print(f"Error searching for ticker: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing ticker data: {e}")
        return None

def get_stock_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        time_series = data.get('Time Series (Daily)')
        if not time_series:
            raise ValueError("No time series data found in the API response")
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in df.columns:
            df[col] = df[col].astype(float)
        one_month_ago = datetime.now() - timedelta(days=30)
        df = df[df.index >= one_month_ago]
        return df
    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching or processing data from Alpha Vantage: {e}")
        return None

def create_stock_chart_with_news(company_name, alpha_vantage_api_key, deduplicated_filtered_articles):
    # Search for the ticker symbol
    ticker = search_ticker(company_name, alpha_vantage_api_key)
    if not ticker:
        return None

    # Get stock data
    stock_data = get_stock_data(ticker, alpha_vantage_api_key)
    if stock_data is None:
        return None

    last_business_day = datetime.now().date()
    while last_business_day.weekday() > 4:  # 5 = Saturday, 6 = Sunday
        last_business_day -= timedelta(days=1)
    
    if stock_data.index.max().date() < last_business_day:
        print(f"Warning: Stock data may not be up-to-date. Last data point: {stock_data.index.max().date()}, Expected: {last_business_day}")

    print(f"Stock data date range: {stock_data.index.min().date()} to {stock_data.index.max().date()}")
    print(f"Number of trading days: {len(stock_data)}")
    print(f"Number of news articles: {len(deduplicated_filtered_articles)}")
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add stock price line
    fig.add_trace(
        go.Scatter(
            x=stock_data.index, 
            y=stock_data['Close'], 
            name=f"{company_name} Stock Price",
            mode='lines',
            line=dict(shape='linear', smoothing=0)
        ),
        secondary_y=False,
    )

    def truncate_text(text, length=85):
        return (text[:length] + '...') if len(text) > length else text

    # Debug information
    print(f"Stock data date range: {stock_data.index.min()} to {stock_data.index.max()}")
    print(f"Number of news articles: {len(deduplicated_filtered_articles)}")

    # Add news bubbles
    news_points_added = 0
    for article in deduplicated_filtered_articles:
        published_date = article.get('date')
        if published_date and published_date != 'Unknown date':
            try:
                date = datetime.strptime(published_date, '%Y-%m-%d').date()
                if date in stock_data.index:
                    nearest_date = date
                else:
                    # Find the nearest trading day
                    nearest_date = min(stock_data.index, key=lambda x: abs(x.date() - date))
                    print(f"Warning: No stock data for {date}, using nearest date {nearest_date.date()}")
                
                price_on_date = stock_data.loc[nearest_date, 'Close']
                jittered_price = price_on_date * (1 + random.uniform(-0.005, 0.005))
                truncated_title = truncate_text(article['title'])
                fig.add_trace(
                    go.Scatter(
                        x=[nearest_date],
                        y=[jittered_price],
                        mode='markers',
                        marker=dict(size=12, symbol='circle', color='red'),
                        name=truncated_title,
                        text=f"Date: {date}<br>Headline: {article['title']}",
                        hoverinfo='text'
                    ),
                    secondary_y=False,
                )
                news_points_added += 1
                print(f"Added news point for date: {date}" + (f" (mapped to: {nearest_date.date()})" if date != nearest_date.date() else ""))
            except ValueError as e:
                print(f"Could not parse date: {published_date}. Error: {e}")
        else:
            print(f"Invalid date for article: {article.get('title', 'Unknown title')}")

    print(f"Total news points added to chart: {news_points_added}")

    # Update layout
    fig.update_layout(
        title_text=f"{company_name} Stock Price with News Events",
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        hovermode="closest",
        width=1200,
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05)
    )

    fig.update_yaxes(title_text="Stock Price (USD)", secondary_y=False)

    return fig