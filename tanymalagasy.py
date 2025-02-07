import requests
from bs4 import BeautifulSoup
import sqlite3
import time
from urllib.parse import urljoin

class DictionaryScraper:
    def __init__(self, base_url, db_path):
        self.base_url = base_url
        self.db_path = db_path
        self.session = requests.Session()
        self.setup_database()

    def setup_database(self):
        """Create the necessary database tables with root words structure"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create root_words table
        c.execute('''
            CREATE TABLE IF NOT EXISTS root_words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root TEXT NOT NULL,
                letter TEXT NOT NULL,
                url TEXT NOT NULL,
                word_type TEXT,
                UNIQUE(root, url)
            )
        ''')
        
        # Create derivatives table
        c.execute('''
            CREATE TABLE IF NOT EXISTS derivatives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root_id INTEGER,
                derivative TEXT NOT NULL,
                url TEXT NOT NULL,
                word_type TEXT,
                FOREIGN KEY (root_id) REFERENCES root_words(id),
                UNIQUE(root_id, derivative)
            )
        ''')
        
        conn.commit()
        conn.close()

    def extract_word_type(self, url):
        """Extract word type from URL (mg.n, mg.av, etc.)"""
        if '#' in url:
            return url.split('#')[1]
        return None

    def scrape_page(self, url, letter):
        """Scrape words from a specific page"""
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Find the word table
        word_table = soup.find('table', {'class': 'menuLink'})
        if not word_table:
            conn.close()
            return
            
        for row in word_table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) != 2:
                continue
                
            # Process root word
            root_link = cells[0].find('a')
            if not root_link:
                continue
                
            root_word = root_link.text.strip()
            root_url = urljoin(self.base_url, root_link['href'])
            root_type = self.extract_word_type(root_url)
            
            # Insert root word
            try:
                c.execute('''
                    INSERT OR IGNORE INTO root_words (root, letter, url, word_type)
                    VALUES (?, ?, ?, ?)
                ''', (root_word, letter, root_url, root_type))
                
                root_id = c.lastrowid or c.execute(
                    'SELECT id FROM root_words WHERE root = ? AND url = ?',
                    (root_word, root_url)
                ).fetchone()[0]
                
                # Process derivatives
                derivatives_cell = cells[1]
                for derivative_link in derivatives_cell.find_all('a'):
                    derivative = derivative_link.text.strip()
                    derivative_url = urljoin(self.base_url, derivative_link['href'])
                    derivative_type = self.extract_word_type(derivative_url)
                    
                    c.execute('''
                        INSERT OR IGNORE INTO derivatives 
                        (root_id, derivative, url, word_type)
                        VALUES (?, ?, ?, ?)
                    ''', (root_id, derivative, derivative_url, derivative_type))
                    
            except sqlite3.Error as e:
                print(f"Error processing word {root_word}: {e}")
                
        conn.commit()
        conn.close()

    def scrape_all_pages(self):
        """Scrape all pages including the default 'A' page"""
        # First scrape the default 'A' page
        print("Scraping letter A (default page)")
        self.scrape_page(self.base_url + "/bins/rootLists", 'A')
        time.sleep(1)  # Be polite
        
        # Get all other letter URLs and scrape them
        response = self.session.get(self.base_url + "/bins/rootLists")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'rootLists?o=let' in href:
                letter = href.split('let')[-1]  # Extract letter from URL
                url = urljoin(self.base_url, href)
                print(f"Scraping letter {letter}")
                self.scrape_page(url, letter.upper())
                time.sleep(1)  # Be polite

def main():
    base_url = "https://motmalgache.org/bins/rootLists"  # The actual base URL
    scraper = DictionaryScraper(base_url, "dictionary.db")
    scraper.scrape_all_pages()

if __name__ == "__main__":
    main()