import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse
import json
import re
from collections import deque

class CNACrawler:
    def __init__(self):
        self.base_url = "https://www.channelnewsasia.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.articles = []

    def _is_valid_article_url(self, url, section=None):
        """
        Validate CNA article URL and optionally constrain to a section path.
        """
        if not url:
            return False

        parsed = urlparse(url.strip())
        path = parsed.path or ''
        hostname = (parsed.netloc or '').lower()

        if hostname and 'channelnewsasia.com' not in hostname:
            return False

        if '/profile/sso/login' in path:
            return False

        if section and not path.startswith(f'/{section}/'):
            return False

        if '-' not in path:
            return False

        return True
        
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

                        full_url = full_url.strip()
                        
                        # Avoid duplicates and non-article pages
                        if full_url not in article_links and self._is_valid_article_url(full_url, section=section):
                            article_links.append(full_url)
                
                time.sleep(1)  # Be respectful with delays
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page {page}: {e}")
                continue
        
        print(f"Found {len(article_links)} article links")
        return article_links

    def get_article_links_from_sitemap(self, section='news', max_links=None):
        """
        Fallback link collection using CNA's news sitemap feed.
        """
        sitemap_url = f"{self.base_url}/api/v1/sitemap-news-feed?_format=xml"
        article_links = []

        try:
            print(f"Fetching sitemap feed for section '{section}'...")
            response = requests.get(sitemap_url, headers=self.headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'xml')
            loc_tags = soup.find_all('loc')

            section_pattern = f"/{section}/"
            for loc in loc_tags:
                url = loc.get_text(strip=True)
                if not url:
                    continue

                if not self._is_valid_article_url(url, section=section):
                    continue

                if url not in article_links:
                    article_links.append(url)

                if max_links and len(article_links) >= max_links:
                    break

            print(f"Found {len(article_links)} sitemap article links for section '{section}'")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching sitemap links: {e}")

        return article_links

    def get_related_article_links(self, seed_links, section='news', max_links=None, max_hops=250):
        """
        Discover additional article links by traversing links found in seed article pages.
        """
        section_pattern = f"/{section}/"
        discovered = list(dict.fromkeys(seed_links))
        discovered_set = set(discovered)
        queue = deque(discovered)
        hops = 0

        print(f"Starting related-link discovery for section '{section}'...")

        while queue and hops < max_hops:
            if max_links and len(discovered) >= max_links:
                break

            current_url = queue.popleft()
            hops += 1

            try:
                response = requests.get(current_url, headers=self.headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    if not href:
                        continue

                    if href.startswith('/'):
                        full_url = urljoin(self.base_url, href)
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue

                    full_url = full_url.strip()

                    if not self._is_valid_article_url(full_url, section=section):
                        continue

                    if full_url in discovered_set:
                        continue

                    discovered.append(full_url)
                    discovered_set.add(full_url)
                    queue.append(full_url)

                    if max_links and len(discovered) >= max_links:
                        break

            except requests.exceptions.RequestException:
                continue

        print(f"Found {len(discovered)} links after related-link discovery")
        return discovered
    
    def scrape_article(self, url):
        """
        Scrape a single article and extract relevant information.
        
        Expected article structure:
        1. Title (h1 tag)
        2. Short description/subtitle (first paragraph after title)
        3. Photo with caption (filtered out)
        4. Location followed by colon (marks start of article content)
        5. Article body (all paragraphs after location)
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
            # First, extract all paragraphs to analyze their classes
            all_paragraphs = soup.find_all('p')
            
            # Separate paragraphs by type: category, description, and content
            category_paragraphs = []
            content_paragraphs = []
            
            for p in all_paragraphs:
                p_class = p.get('class', [])
                p_text = p.get_text(separator=' ', strip=True)
                
                if not p_text:  # Skip empty paragraphs
                    continue
                    
                # Check if it's a category paragraph
                if any('category' in cls.lower() for cls in p_class):
                    category_paragraphs.append(p_text)
                else:
                    content_paragraphs.append(p_text)
            
            # For backwards compatibility, use content_paragraphs as body_paragraphs
            body_paragraphs = content_paragraphs
            
            # Parse body structure based on CNA article format:
            # 1. Title (already extracted)
            # 2. Category paragraphs (e.g., "Singapore", "World") - extracted from p tags with 'category' class
            # 3. Short description (first non-category paragraph)
            # 4. Photo caption (to be filtered out)
            # 5. Location followed by colon
            # 6. Article content
            
            section = None
            subtitle = None
            description = None
            location = None
            clean_paragraphs = []
            article_started = False
            
            # Extract section from category paragraphs
            if category_paragraphs:
                # Use the first category paragraph as the section
                section = category_paragraphs[0].strip()
            
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
            
            # Photo caption patterns to filter
            # Captions often mention names of people, dates, and photo-related terms
            caption_patterns = [
                r'\((File )?[Pp]hoto:\s*[^)]+\)',  # (Photo: ...) or (File photo: ...)
                r'\([Ff]ile [Pp]hoto\)',            # (File Photo)
                r'\([Ii]mage:\s*[^)]+\)',           # (Image: ...)
                r'^Photo:\s*.*$',                    # Standalone "Photo: ..." line
                r'^File [Pp]hoto:\s*.*$'            # Standalone "File photo: ..." line
            ]
            
            # Identify photo captions by content patterns
            def is_likely_caption(text):
                """Heuristic to identify photo captions based on content."""
                # Skip AI audio disclaimer
                if 'This audio is generated by an AI tool' in text:
                    return True
                    
                # Check explicit caption patterns
                for pattern in caption_patterns:
                    if re.search(pattern, text):
                        return True
                
                # Captions often describe people in photos with dates/speaking/locations
                # e.g., "MPs Saktiandi Supaat, David Hoe and Yip Hon Weng speaking in parliament on Feb 24, 2026."
                caption_indicators = [
                    r'speaking (in|at|on)',
                    r'\b(on|dated)\s+[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}',  # Date patterns
                    r'^[A-Z][\w\s,]+\s+(speaking|attending|at|in)\s+',  # Names + action
                ]
                
                # Short paragraphs mentioning people and dates are likely captions
                if len(text) < 150:
                    for indicator in caption_indicators:
                        if re.search(indicator, text):
                            return True
                            
                return False
            
            if body_paragraphs:
                # Find the first non-caption paragraph as the description
                description_found = False
                
                for i, para in enumerate(body_paragraphs):
                    para = para.strip()
                    
                    if not para:
                        continue
                    
                    # Check if this is a photo caption
                    if is_likely_caption(para):
                        continue
                    
                    # Check for location prefix (e.g., "SINGAPORE:", "TUCSON, Arizona:")
                    # Location marks the start of the article content
                    location_match = re.match(r'^([A-Z][A-Z\s,\-]+):\s*(.*)$', para)
                    if location_match and not article_started:
                        location = location_match.group(1).strip()
                        remaining_text = location_match.group(2).strip()
                        article_started = True  # Article content starts here
                        
                        # Add remaining text if present
                        if remaining_text:
                            clean_paragraphs.append(remaining_text)
                        continue
                    
                    # If article hasn't started yet and this is the first non-caption paragraph,
                    # it's the description
                    if not article_started and not description_found:
                        description = para
                        description_found = True
                        continue
                    
                    # If article has started, collect content
                    if article_started:
                        # Skip newsletter/subscription prompts
                        if any(keyword in para for keyword in newsletter_keywords):
                            break
                        
                        # Add clean paragraph to body content
                        clean_paragraphs.append(para)
                
                # Set subtitle as description for backwards compatibility
                subtitle = description
            
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
                'description': description,  # Short description after title
                'subtitle': subtitle,        # Alias for description (backwards compatibility)
                'location': location,        # Location where article was written
                'body_content': ' '.join(clean_paragraphs),  # Main article content after location
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
        self.articles = []
        
        # Get article links
        article_links = self.get_article_links(section, max_pages)

        # Fallback to sitemap feed if listing pages do not provide enough links.
        if max_articles and len(article_links) < max_articles:
            needed = max_articles - len(article_links)
            print(f"Need {needed} more links. Trying sitemap fallback...")
            sitemap_links = self.get_article_links_from_sitemap(section, max_links=max_articles * 3)
            for link in sitemap_links:
                if link not in article_links:
                    article_links.append(link)
                if len(article_links) >= max_articles:
                    break

            print(f"Total links after sitemap fallback: {len(article_links)}")

        if max_articles and len(article_links) < max_articles:
            needed = max_articles - len(article_links)
            print(f"Need {needed} more links. Trying related-link discovery...")
            related_links = self.get_related_article_links(
                article_links,
                section=section,
                max_links=max_articles * 3,
            )
            for link in related_links:
                if link not in article_links:
                    article_links.append(link)
                if len(article_links) >= max_articles:
                    break

            print(f"Total links after related-link discovery: {len(article_links)}")
        
        # Scrape each article
        for i, url in enumerate(article_links, 1):
            if max_articles and len(self.articles) >= max_articles:
                break

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
        desc = articles[0].get('description', '')
        if desc:
            print(f"Description: {desc[:150]}..." if len(desc) > 150 else f"Description: {desc}")
        print(f"Location: {articles[0]['location']}")
        print(f"Category: {articles[0]['category']}")
        print(f"Date: {articles[0]['publication_date']}")
        print(f"Content preview: {articles[0]['body_content'][:250]}...")
        print(f"Total content length: {len(articles[0]['body_content'])} characters")
