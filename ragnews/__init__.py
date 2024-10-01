#!/bin/python3

'''
Run an interactive QA session with the news articles using the Groq LLM API and retrieval augmented generation (RAG).

New articles can be added to the database with the --add_url parameter,
and the path to the database can be changed with the --db parameter.
'''

from bs4 import BeautifulSoup
from urllib.parse import urlparse
import datetime
import logging
import re
import requests
import sqlite3
import time
import random

from groq import Groq, RateLimitError, InternalServerError
import metahtml

import os


################################################################################
# LLM functions
################################################################################

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def run_llm(system, user, model='llama-3.1-8b-instant', seed=None, temperature=None, stop=None, verbose=False):
    """
    Helper function to interact with an LLM.
    Handles system and user prompts and supports retries with backoff.
    """

    if verbose:
        logging.info(f"Sending system prompt to LLM: {system[:100]}...")
        logging.info(f"Sending user prompt to LLM: {user[:100]}...")

    # Simulate delay to avoid rapid-fire requests
    time.sleep(3)

    try:
        # Perform API call with retry logic
        chat_completion = execute_with_retry(
            messages=[
                {
                    'role': 'system',
                    'content': system,
                },
                {
                    "role": "user",
                    "content": user,
                }
            ],
            model=model,
            seed=seed,
            temperature=temperature,
            stop=stop,
        )
        
        result = chat_completion.choices[0].message.content

        if verbose:
            logging.info(f"LLM result: {result[:500]}...")

        return result

    except Exception as e:
        logging.error(f"LLM call failed: {str(e)}")
        return "Error: LLM call failed"

def summarize_text(text, seed=None):
    system = 'Summarize the input text below.  Limit the summary to 1 paragraph.  Use an advanced reading level similar to the input text, and ensure that all people, places, and other proper and dates nouns are included in the summary.  The summary should be in English.'
    return run_llm(system, text, seed=seed)


def translate_text(text):
    system = 'You are a professional translator working for the United Nations.  The following document is an important news article that needs to be translated into English.  Provide a professional translation.'
    return run_llm(system, text)


def extract_keywords(text, seed=None):
    r'''
    This is a helper function for RAG.
    Given an input text,
    this function extracts the keywords that will be used to perform the search for articles that will be used in RAG.

    >>> extract_keywords('Who is the current democratic presidential nominee?', seed=0)
    'Democratic nominee candidate Joe Biden president election politics'
    >>> extract_keywords('What is the policy position of Trump related to illegal Mexican immigrants?', seed=0)
    'Illegal immigration Trump border wall Mexico deportation ICE enforcement visa security national securityylum seeker refugees'

    Note that the examples above are passing in a seed value for deterministic results.
    In production, you probably do not want to specify the seed.
    '''

    system = '''Respond with exactly ten search keywords from and related to the input below. Do not attempt to answer questions using the keywords. Stay focused on providing keywords that reflect the main concepts and topics of the input. Avoid using compound words or names unless they are essential. Return the keywords in a single space-separated list of exactly 10 words. Do not include new lines, bullet points, or punctuation.'''
    return run_llm(system, text, seed=seed)


################################################################################
# helper functions
################################################################################

def _logsql(sql):
    rex = re.compile(r'\W+')
    sql_dewhite = rex.sub(' ', sql)
    logging.debug(f'SQL: {sql_dewhite}')


def _catch_errors(func):
    '''
    This function is intended to be used as a decorator.
    It traps whatever errors the input function raises and logs the errors.
    We use this decorator on the add_urls method below to ensure that a webcrawl continues even if there are errors.
    '''
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.error(str(e))
    return inner_function

def retry_with_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (RateLimitError, InternalServerError),
):
    """
    Retry a function with exponential backoff.

    Args:
        func (callable): The function to retry.
        initial_delay (float): The initial delay before retrying.
        exponential_base (float): The base for the exponential backoff.
        jitter (bool): Whether to add randomness to the delay.
        max_retries (int): The maximum number of retries.
        errors (tuple): The exceptions to catch for retrying.

    Returns:
        callable: A wrapped function that retries with exponential backoff.
    """
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"warning: rate limited, slowing down... {delay} seconds")
                time.sleep(delay)
            except Exception as e:
                raise e

    return wrapper

@retry_with_backoff
def execute_with_retry(**kwargs):
    """
    Make a hypothetical API call with retry logic.

    Args:
        **kwargs: Arbitrary keyword arguments for the API call.

    Returns:
        dict: The API response.
    """
    # Simulate API call or logic that may fail
    return client.chat.completions.create(**kwargs)

################################################################################
# rag
################################################################################


def rag(text, db, keywords_text=None, system=None, temperature=None, stop=None, max_articles_length=20000, verbose=False):
    '''
    This function uses retrieval augmented generation (RAG) to generate an LLM response to the input text.
    The db argument should be an instance of the `ArticleDB` class that contains the relevant documents to use.

    NOTE:
    There are no test cases because:
    1. the answers are non-deterministic (both because of the LLM and the database), and
    2. evaluating the quality of answers automatically is non-trivial.

    '''
    if keywords_text is None:
        logging.info(f"Keywords not provided. Extracting keywords for text: {text}")
        keywords_text = extract_keywords(text, seed=0)

    articles = db.find_articles(keywords_text, limit=5)
    if len(articles) == 0:
        return "No articles found"
    if max_articles_length is not None:
        # Calculate the combined length of all articles
        combined_length = sum(
            len(article.get('en_translation', article.get('text', '')))  # Use get() with fallback to avoid KeyError
            for article in articles
        )
        logging.info(f'combined_length: {combined_length}')
        
        # Remove articles until the combined length is less than max_articles_length
        while combined_length > max_articles_length and articles:
            removed_article = articles.pop()
            combined_length -= len(
                removed_article.get('en_translation', removed_article.get('text', ''))  # Use get() with fallback
            )
            logging.info(f"Removed article. New combined length: {combined_length}")

    article_template = (
        "<article>\n"
        "  <title>{title}</title>\n"
        "  <publish_date>{publish_date}</publish_date>\n"
        "  <text>{text}</text>\n"
        "  <source>{url}</source>\n"
        "</article>\n"
    )
    articles_str = '\n'.join([
        article_template.format(
            title=article['title'],
            publish_date=article['publish_date'],
            text=article['en_translation'] if article.get('en_translation') is not None else article['text'],
            url=article['url']
        ) for article in articles
    ])

    if system is None:
        system = "You are a helpful research assistant that tries to answer the user's question based on the information provided from articles below. Do not ever use any information outside of the articles provided. Only respond with the answer to the question. Use numbered citations like [1] [2] [3] at the end of sentences. Always respond with the full citation at the end of the answer like this:\nSources:\n[1] https://example.com/article1\n[2] https://example.com/article2\n[3] https://example.com/article3"
    user = (
        "<context>\n"
        "{articles_str}\n"
        "</context>\n"
        "<question>\n"
        "{text}\n"
        "</question>\n"
    ).format(articles_str=articles_str, text=text)
    return run_llm(system, user, temperature=temperature, stop=stop, verbose=verbose)

class ArticleDB:
    '''
    This class represents a database of news articles.
    It is backed by sqlite3 and designed to have no external dependencies and be easy to understand.

    The following example shows how to add urls to the database.

    >>> db = ArticleDB()
    >>> len(db)
    0
    
    Once articles have been added,
    we can search through those articles to find articles about only certain topics.
    
    >>> articles = db.find_articles('Economía')

    The output is a list of articles that match the search query.
    Each article is represented by a dictionary with a number of fields about the article.

    '''

    _TESTURLS = [
        'https://elpais.com/economia/2024-09-06/la-creacion-de-empleo-defrauda-en-estados-unidos-en-agosto-y-aviva-el-fantasma-de-la-recesion.html',
        'https://www.cnn.com/2024/09/06/politics/american-push-israel-hamas-deal-analysis/index.html',
        ]

    def __init__(self, filename=':memory:'):
        self.db = sqlite3.connect(filename)
        self.db.row_factory=sqlite3.Row
        self.logger = logging
        self._create_schema()

    def _create_schema(self):
        '''
        Create the DB schema if it doesn't already exist.

        The test below demonstrates that creating a schema on a database that already has the schema will not generate errors.

        >>> db = ArticleDB()
        >>> db._create_schema()
        >>> db._create_schema()
        '''
        try:
            sql = '''
            CREATE VIRTUAL TABLE articles
            USING FTS5 (
                title,
                text,
                hostname,
                url,
                publish_date,
                crawl_date,
                lang,
                en_translation,
                en_summary
                );
            '''
            self.db.execute(sql)
            self.db.commit()

        # if the database already exists,
        # then do nothing
        except sqlite3.OperationalError:
            self.logger.debug('CREATE TABLE failed')

    def find_articles(self, query, limit=10, timebias_alpha=1):
        '''
        Return a list of articles in the database that match the specified query.

        Lowering the value of the timebias_alpha parameter will result in the time becoming more influential.
        The final ranking is computed by the FTS5 rank * timebias_alpha / (days since article publication + timebias_alpha).
        '''
        # SQL query to fetch matching articles using FTS5 with time-adjusted ranking
        sql = ''' 
        SELECT title, text, publish_date, url, en_summary,
        rank,
        rank * ? / (julianday('now') - julianday(publish_date) + ?) AS time_adjusted_rank
        FROM articles
        WHERE articles MATCH ?
        ORDER BY ABS(time_adjusted_rank) DESC
        LIMIT ?;
        '''
        
        query = re.sub(r'[^\w\s]', '', query, flags=re.UNICODE)
        
        _logsql(sql)
        cursor = self.db.cursor()
        try:
            cursor.execute(sql, (timebias_alpha, timebias_alpha, query, limit))
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logging.error(f"SQLite error during query: {e}. Query: {query}")
            return []

        # Process the results into a list of dictionaries
        articles = []
        for row in rows:
            article = {
                'title': row['title'],
                'text': row['text'],
                'publish_date': row['publish_date'],
                'url': row['url'],
                'en_summary': row['en_summary'],
            }
            articles.append(article)
    
        return articles
        
    @_catch_errors
    def add_url(self, url, recursive_depth=0, allow_dupes=False):
        '''
        Download the url, extract various metainformation, and add the metainformation into the db.

        By default, the same url cannot be added into the database multiple times.

        >>> db = ArticleDB()
        >>> db.add_url(ArticleDB._TESTURLS[0])
        >>> db.add_url(ArticleDB._TESTURLS[0])
        >>> db.add_url(ArticleDB._TESTURLS[0])
        >>> len(db)
        1

        >>> db = ArticleDB()
        >>> db.add_url(ArticleDB._TESTURLS[0], allow_dupes=True)
        >>> db.add_url(ArticleDB._TESTURLS[0], allow_dupes=True)
        >>> db.add_url(ArticleDB._TESTURLS[0], allow_dupes=True)
        >>> len(db)
        3

        '''
        logging.info(f'add_url {url}')

        if not allow_dupes:
            logging.debug(f'checking for url in database')
            sql = '''
            SELECT count(*) FROM articles WHERE url=?;
            '''
            _logsql(sql)
            cursor = self.db.cursor()
            cursor.execute(sql, [url])
            row = cursor.fetchone()
            is_dupe = row[0] > 0
            if is_dupe:
                logging.debug(f'duplicate detected, skipping!')
                return

        logging.debug(f'downloading url')
        try:
            response = requests.get(url)
        except requests.exceptions.MissingSchema:
            # if no schema was provided in the url, add a default
            url = 'https://' + url
            response = requests.get(url)
        parsed_uri = urlparse(url)
        hostname = parsed_uri.netloc

        logging.debug(f'extracting information')
        parsed = metahtml.parse(response.text, url)
        info = metahtml.simplify_meta(parsed)

        if info['type'] != 'article' or len(info['content']['text']) < 100:
            logging.debug(f'not an article... skipping')
            en_translation = None
            en_summary = None
            info['title'] = None
            info['content'] = {'text': None}
            info['timestamp.published'] = {'lo': None}
            info['language'] = None
        else:
            logging.debug('summarizing')
            if not info['language'].startswith('en'):
                en_translation = translate_text(info['content']['text'])
            else:
                en_translation = None
            en_summary = summarize_text(info['content']['text'])

        logging.debug('inserting into database')
        sql = '''
        INSERT INTO articles(title, text, hostname, url, publish_date, crawl_date, lang, en_translation, en_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql, [
            info['title'],
            info['content']['text'], 
            hostname,
            url,
            info['timestamp.published']['lo'],
            datetime.datetime.now().isoformat(),
            info['language'],
            en_translation,
            en_summary,
            ])
        self.db.commit()

        logging.debug('recursively adding more links')
        if recursive_depth > 0:
            for link in info['links.all']:
                url2 = link['href']
                parsed_uri2 = urlparse(url2)
                hostname2 = parsed_uri2.netloc
                if hostname in hostname2 or hostname2 in hostname:
                    self.add_url(url2, recursive_depth-1)
        
    def __len__(self):
        sql = '''
        SELECT count(*)
        FROM articles
        WHERE text IS NOT NULL;
        '''
        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        return row[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loglevel', default='warning')
    parser.add_argument('--db', default='ragnews.db')
    parser.add_argument('--recursive_depth', default=0, type=int)
    parser.add_argument('--query', help='Query to run against the RAG system')
    parser.add_argument('--add_url', help='If this parameter is added, then the program will not provide an interactive QA session with the database.  Instead, the provided url will be downloaded and added to the database.')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=args.loglevel.upper(),
        )

    db = ArticleDB(args.db)

    if args.add_url:
        db.add_url(args.add_url, recursive_depth=args.recursive_depth, allow_dupes=True)

    elif args.query:
        response = rag(args.query, db)
        print(response)

    else:
        import readline
        while True:
            text = input('ragnews> ')
            if len(text.strip()) > 0:
                output = rag(text, db)
                print(output)
