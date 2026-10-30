"""
Database connection, schema management, and persistence layer using SQLite.
Ensures ACID compliance, transactional integrity, and normalized schema storage.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from src.config import config


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a connection to the SQLite database with WAL mode enabled."""
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables with indexes."""
    conn = get_connection(db_path)
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS assets (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS price_bars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            timestamp_est TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            dividends REAL DEFAULT 0.0,
            stock_splits REAL DEFAULT 0.0,
            FOREIGN KEY (symbol) REFERENCES assets(symbol),
            UNIQUE(symbol, timestamp_utc)
        );

        CREATE INDEX IF NOT EXISTS idx_price_bars_symbol_time ON price_bars(symbol, timestamp_utc);

        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id TEXT UNIQUE NOT NULL,
            symbol TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            timestamp_est TEXT NOT NULL,
            author_username TEXT,
            author_followers INTEGER DEFAULT 0,
            text TEXT NOT NULL,
            likes INTEGER DEFAULT 0,
            reposts INTEGER DEFAULT 0,
            replies INTEGER DEFAULT 0,
            views INTEGER DEFAULT 0,
            sentiment_score REAL DEFAULT 0.0,
            sentiment_subjectivity REAL DEFAULT 0.0,
            engagement_weight REAL DEFAULT 1.0,
            FOREIGN KEY (symbol) REFERENCES assets(symbol)
        );

        CREATE INDEX IF NOT EXISTS idx_tweets_symbol_time ON tweets(symbol, timestamp_utc);

        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id TEXT UNIQUE NOT NULL,
            symbol TEXT NOT NULL,
            published_at_utc TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            snippet TEXT,
            sentiment_score REAL DEFAULT 0.0,
            source TEXT DEFAULT 'tavily',
            FOREIGN KEY (symbol) REFERENCES assets(symbol)
        );

        CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news_articles(symbol, published_at_utc);

        CREATE TABLE IF NOT EXISTS model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            predicted_price REAL NOT NULL,
            actual_price REAL,
            predicted_return REAL NOT NULL,
            actual_return REAL,
            model_name TEXT NOT NULL,
            fold_index INTEGER NOT NULL,
            FOREIGN KEY (symbol) REFERENCES assets(symbol)
        );

        CREATE TABLE IF NOT EXISTS strategy_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            position INTEGER NOT NULL, -- -1 (Short), 0 (Flat), 1 (Long)
            asset_price REAL NOT NULL,
            portfolio_value REAL NOT NULL,
            strategy_return REAL NOT NULL,
            benchmark_return REAL NOT NULL,
            drawdown REAL NOT NULL
        );
        """)
    conn.close()


def save_price_bars(df: pd.DataFrame, symbol: str, db_path: Optional[Path] = None) -> int:
    """Upsert price bars into SQLite."""
    if df.empty:
        return 0
    conn = get_connection(db_path)
    records = []
    for _, row in df.iterrows():
        records.append((
            symbol,
            str(row["timestamp_utc"]),
            str(row["timestamp_est"]),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            float(row["Volume"]),
            float(row.get("Dividends", 0.0)),
            float(row.get("Stock Splits", 0.0))
        ))
    with conn:
        conn.execute("INSERT OR IGNORE INTO assets (symbol, name, asset_type) VALUES (?, ?, ?)",
                     (symbol, symbol, "equity"))
        conn.executemany("""
            INSERT OR REPLACE INTO price_bars 
            (symbol, timestamp_utc, timestamp_est, open, high, low, close, volume, dividends, stock_splits)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
    conn.close()
    return len(records)


def save_tweets(df: pd.DataFrame, symbol: str, db_path: Optional[Path] = None) -> int:
    """Upsert tweets into SQLite."""
    if df.empty:
        return 0
    conn = get_connection(db_path)
    records = []
    for _, row in df.iterrows():
        records.append((
            str(row["tweet_id"]),
            symbol,
            str(row["timestamp_utc"]),
            str(row["timestamp_est"]),
            str(row.get("author_username", "anonymous")),
            int(row.get("author_followers", 0)),
            str(row["text"]),
            int(row.get("likes", 0)),
            int(row.get("reposts", 0)),
            int(row.get("replies", 0)),
            int(row.get("views", 0)),
            float(row.get("sentiment_score", 0.0)),
            float(row.get("sentiment_subjectivity", 0.0)),
            float(row.get("engagement_weight", 1.0))
        ))
    with conn:
        conn.execute("INSERT OR IGNORE INTO assets (symbol, name, asset_type) VALUES (?, ?, ?)",
                     (symbol, symbol, "equity"))
        conn.executemany("""
            INSERT OR REPLACE INTO tweets
            (tweet_id, symbol, timestamp_utc, timestamp_est, author_username, author_followers,
             text, likes, reposts, replies, views, sentiment_score, sentiment_subjectivity, engagement_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
    conn.close()
    return len(records)


def save_news(df: pd.DataFrame, symbol: str, db_path: Optional[Path] = None) -> int:
    """Upsert news articles into SQLite."""
    if df.empty:
        return 0
    conn = get_connection(db_path)
    records = []
    for _, row in df.iterrows():
        records.append((
            str(row["article_id"]),
            symbol,
            str(row["published_at_utc"]),
            str(row["title"]),
            str(row.get("url", "")),
            str(row.get("snippet", "")),
            float(row.get("sentiment_score", 0.0)),
            str(row.get("source", "tavily"))
        ))
    with conn:
        conn.execute("INSERT OR IGNORE INTO assets (symbol, name, asset_type) VALUES (?, ?, ?)",
                     (symbol, symbol, "equity"))
        conn.executemany("""
            INSERT OR REPLACE INTO news_articles
            (article_id, symbol, published_at_utc, title, url, snippet, sentiment_score, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
    conn.close()
    return len(records)


def load_price_bars(symbol: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load price bars ordered chronologically."""
    conn = get_connection(db_path)
    query = """
        SELECT timestamp_utc, timestamp_est, open, high, low, close, volume
        FROM price_bars
        WHERE symbol = ?
        ORDER BY timestamp_utc ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["timestamp_est"] = pd.to_datetime(df["timestamp_est"], utc=True).dt.tz_convert("America/New_York")
    return df


def load_tweets(symbol: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load tweets ordered chronologically."""
    conn = get_connection(db_path)
    query = """
        SELECT tweet_id, timestamp_utc, timestamp_est, author_username, author_followers,
               text, likes, reposts, replies, views, sentiment_score, sentiment_subjectivity,
               engagement_weight
        FROM tweets
        WHERE symbol = ?
        ORDER BY timestamp_utc ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["timestamp_est"] = pd.to_datetime(df["timestamp_est"], utc=True).dt.tz_convert("America/New_York")
    return df


def load_news(symbol: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load news articles."""
    conn = get_connection(db_path)
    query = """
        SELECT article_id, published_at_utc, title, url, snippet, sentiment_score, source
        FROM news_articles
        WHERE symbol = ?
        ORDER BY published_at_utc ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    if not df.empty:
        df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], utc=True)
    return df
