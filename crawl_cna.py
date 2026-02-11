import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
from urllib.parse import urljoin
import json
import re

class CNACrawler:
    def __init__(self):
        self.base_url = "https://www.channelnewsasia.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.articles = []
        
    def get_article_links(self, section='news', max_pages=1):
        """
        Get article links from a specific section.
        Sections: 'news', 'business', 'sport', 'asia', 'singapore', 'world'
        """
        article_links = []
        
        for page in range(1, max_pages + 1):
            url = f"{self.base_url}/{section}"
            if page > 1:
                url += f"?page={page}"
            
            try:
                print(f"Fetching page {page} from {section}...")
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links - CNA uses various classes
                # Look for article links in common patterns
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link.get('href', '')
                    # Filter for article URLs (typically contain year pattern)
                    if any(x in href for x in ['/news/', '/singapore/', '/asia/', '/world/', '/business/', '/sport/']):
                        if href.startswith('/'):
                            full_url = urljoin(self.base_url, href)
                        elif href.startswith('http'):
                            full_url = href
                        else:
                            continue
                        
                        # Avoid duplicates and non-article pages
                        if full_url not in article_links and '-' in href:
                            article_links.append(full_url)
                
                time.sleep(1)  # Be respectful with delays
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page {page}: {e}")
                continue
        
        print(f"Found {len(article_links)} article links")
        return article_links
    
    def scrape_article(self, url):
        """
        Scrape a single article and extract relevant information.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = None
            title_tag = soup.find('h1')
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            # Extract article body - CNA uses different content containers
            body_paragraphs = []
            
            # Try different selectors for article content
            content_containers = [
                soup.find('div', class_='text-long'),
                soup.find('div', class_='article__body'),
                soup.find('article'),
                soup.find('div', {'class': lambda x: x and 'content' in x.lower() if x else False})
            ]
            
            for container in content_containers:
                if container:
                    paragraphs = container.find_all('p')
                    body_paragraphs = [p.get_text(separator=' ', strip=True) for p in paragraphs if p.get_text(strip=True)]
                    if body_paragraphs:
                        break
            
            # Parse body structure and extract metadata
            section = None
            subtitle = None
            location = None
            clean_paragraphs = []
            
            # Newsletter/subscription keywords to filter out at the end
            newsletter_keywords = [
                "Get our pick of top stories",
                "Stay updated with notifications",
                "Get WhatsApp alerts",
                "Join our channel for the top reads",
                "Subscribe to",
                "This service is not intended for persons residing in the E.U.",
                "By clicking subscribe"
            ]
            
            if body_paragraphs:
                # First paragraph often contains: [SECTION] [Subtitle]
                first_para = body_paragraphs[0]
                
                # Try to identify section at the beginning (Singapore, World, Asia, etc.)
                section_keywords = ['Singapore', 'World', 'Asia', 'Business', 'Sport']
                for keyword in section_keywords:
                    if first_para.startswith(keyword + ' '):
                        section = keyword
                        # Remove section from first paragraph to get subtitle
                        subtitle = first_para[len(keyword):].strip()
                        break
                
                if not subtitle and first_para:
                    subtitle = first_para
                
                # Process remaining paragraphs
                start_index = 1
                for i, para in enumerate(body_paragraphs[start_index:], start=start_index):
                    # Remove photo captions from paragraph (including "File photo:" variations)
                    para = re.sub(r'\((File )?[Pp]hoto:\s*[^)]+\)', '', para)
                    para = para.strip()
                    
                    # Skip if paragraph is now empty after removing caption
                    if not para:
                        continue
                    
                    # Skip AI audio disclaimer
                    if 'This audio is generated by an AI tool' in para:
                        continue
                    
                    # Check for location prefix (e.g., "SINGAPORE:", "TUCSON, Arizona:")
                    location_match = re.match(r'^([A-Z][A-Z\s,]+):\s*(.+)$', para)
                    if location_match and not location:
                        location = location_match.group(1).strip()
                        remaining_text = location_match.group(2).strip()
                        if remaining_text:
                            clean_paragraphs.append(remaining_text)
                        continue
                    
                    # Skip newsletter/subscription prompts
                    if any(keyword in para for keyword in newsletter_keywords):
                        break
                    
                    # Add clean paragraph to body content
                    clean_paragraphs.append(para)
            
            # Extract publication date - try multiple selectors
            pub_date = None
            time_tag = soup.find('time')
            if time_tag:
                pub_date = time_tag.get('datetime') or time_tag.get_text(strip=True)
            
            # If not found, try other date selectors
            if not pub_date:
                date_elements = soup.find_all(['span', 'div'], class_=lambda x: x and 'date' in x.lower() if x else False)
                for elem in date_elements:
                    date_text = elem.get_text(strip=True)
                    if date_text:
                        pub_date = date_text
                        break
            
            # Extract category from URL if not found in breadcrumb
            category = None
            breadcrumb = soup.find('nav', class_='breadcrumb')
            if breadcrumb:
                links = breadcrumb.find_all('a')
                if links:
                    category = links[-1].get_text(strip=True)
            
            # Fallback: extract category from URL
            if not category:
                url_parts = url.split('/')
                for part in url_parts:
                    if part in ['singapore', 'world', 'asia', 'business', 'sport', 'news']:
                        category = part.title()
                        break
            
            # Use section as category fallback
            if not category and section:
                category = section
            
            article_data = {
                'url': url,
                'title': title,
                'section': section,
                'subtitle': subtitle,
                'location': location,
                'body_content': ' '.join(clean_paragraphs),
                'publication_date': pub_date,
                'category': category,
                'scraped_at': datetime.now().isoformat()
            }
            
            return article_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error scraping article {url}: {e}")
            return None
    
    def crawl(self, section='news', max_pages=1, max_articles=None):
        """
        Main crawling function.
        """
        print(f"Starting crawl of CNA {section} section...")
        
        # Get article links
        article_links = self.get_article_links(section, max_pages)
        
        # Limit number of articles if specified
        if max_articles:
            article_links = article_links[:max_articles]
        
        # Scrape each article
        for i, url in enumerate(article_links, 1):
            print(f"Scraping article {i}/{len(article_links)}: {url}")
            
            article = self.scrape_article(url)
            if article and article['body_content']:  # Only add if body content exists
                self.articles.append(article)
                loc_info = f" [{article['location']}]" if article['location'] else ""
                print(f"  ✓ Scraped: {article['title'][:50]}...{loc_info}")
            else:
                print(f"  ✗ Failed to scrape or no content")
            
            time.sleep(2)  # Be respectful with delays
        
        print(f"\nCrawling complete! Collected {len(self.articles)} articles.")
        return self.articles
    
    def save_to_csv(self, filename='cna_articles.csv'):
        """Save articles to CSV file."""
        if not self.articles:
            print("No articles to save!")
            return
        
        df = pd.DataFrame(self.articles)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved {len(self.articles)} articles to {filename}")
    
    def save_to_json(self, filename='cna_articles.json'):
        """Save articles to JSON file."""
        if not self.articles:
            print("No articles to save!")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.articles)} articles to {filename}")


# Example Usage
if __name__ == "__main__":
    crawler = CNACrawler()
    
    # Crawl news section - adjust parameters as needed
    articles = crawler.crawl(
        section='news',      # Options: 'news', 'singapore', 'asia', 'world', 'business', 'sport'
        max_pages=2,         # Number of listing pages to crawl
        max_articles=10      # Maximum number of articles to scrape (None for all)
    )
    
    # Save results
    crawler.save_to_csv('cna_articles.csv')
    crawler.save_to_json('cna_articles.json')
    
    # Display sample
    if articles:
        print("\n" + "="*80)
        print("Sample Article:")
        print("="*80)
        print(f"Title: {articles[0]['title']}")
        print(f"Section: {articles[0]['section']}")
        print(f"Subtitle: {articles[0]['subtitle'][:100]}..." if articles[0]['subtitle'] and len(articles[0]['subtitle']) > 100 else f"Subtitle: {articles[0]['subtitle']}")
        print(f"Location: {articles[0]['location']}")
        print(f"Category: {articles[0]['category']}")
        print(f"Date: {articles[0]['publication_date']}")
        print(f"Content preview: {articles[0]['body_content'][:200]}...")
