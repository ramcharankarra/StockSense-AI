# StockSense AI — High-Level Technical Architecture

StockSense AI is built with a decoupled, highly modular three-tier financial technology architecture. This design completely segregates data ingestion and quantitative model execution from the Streamlit UI presentation layer.

---

## 🏗️ System Components

The platform is structured into three distinct execution layers:

```
+-----------------------------------------------------------------------+
|                       PRESENTATION LAYER (UI)                         |
|      (Streamlit Web Application Panels & Plotly Chart Containers)      |
+-----------------------------------------------------------------------+
                                  |
                                  | Session State / Dynamic Reruns
                                  v
+-----------------------------------------------------------------------+
|                    COMPUTATIONAL ENGINE (MIDDLEWARE)                  |
|  (Ensemble ML Pipeline, Math Risk Engines, Sentiment Scoring, Signals) |
+-----------------------------------------------------------------------+
                                  |
                                  | ORM / Parameterized Queries
                                  v
+-----------------------------------------------------------------------+
|                         DATA & STORAGE LAYER                          |
|         (yfinance Market Feed, SQLite / PostgreSQL Database)          |
+-----------------------------------------------------------------------+
```

---

## 🗄️ Relational Database Schema Blueprint

The platform implements a highly optimized database architecture with index structures on frequently accessed coordinates (such as stock symbols and query timestamps). It is designed to run on a local thread-safe **SQLite** environment during development and can be scaled to a **PostgreSQL** cluster by modifying the `DATABASE_URL` inside the `.env` configuration file without modifying a single line of Python code.

```
       +--------------------+
       |       users        |
       |  (PK) id           | <--------------------+
       |       username     |                      |
       |       email        |                      |
       |       password     |                      |
       |       created_at   |                      |
       +--------------------+                      |
                 |                                 | (1 : 1)
                 | (1 : N)                         |
                 v                                 v
       +--------------------+            +--------------------+
       |     portfolios     |            |   user_settings    |
       |  (PK) id           |            |  (PK) user_id (FK) |
       |  (FK) user_id      |            |       default_tick |
       |       name         |            |       default_period|
       |       description  |            |       risk_free_rate|
       +--------------------+            |       watchlist    |
                 |                       +--------------------+
                 | (1 : N)
                 v
       +--------------------+
       |      holdings      |
       |  (PK) id           |
       |  (FK) portfolio_id |
       |       symbol       |
       |       shares       |
       |       avg_cost     |
       +--------------------+

  ====================== Caching & Telemetry Tables ======================

  +--------------------------------------------------------------------+
  |  stock_prices (PK: symbol + date)                                  |
  |  - Tracks: Date, Open, High, Low, Close, Adj_Close, Volume          |
  +--------------------------------------------------------------------+
  
  +--------------------------------------------------------------------+
  |  indicators (PK: symbol + date)                                    |
  |  - Tracks: RSI, MACD, MACD_Signal, BB_Upper, BB_Lower, SMA50, SMA200|
  +--------------------------------------------------------------------+
  
  +--------------------------------------------------------------------+
  |  predictions (PK: id, FK: symbol)                                  |
  |  - Tracks: Model_Name, Horizon_Days, Predicted_Price, RMSE, MAPE   |
  +--------------------------------------------------------------------+
  
  +--------------------------------------------------------------------+
  |  sentiments (PK: id, FK: symbol)                                   |
  |  - Tracks: Overall_Score, Sentiment_Label, Article_Count, Timestamp|
  +--------------------------------------------------------------------+
  
  +--------------------------------------------------------------------+
  |  backtests (PK: id, FK: user_id)                                   |
  |  - Tracks: Strategy, Symbol, Total_Return, Sharpe, Max_Drawdown    |
  +--------------------------------------------------------------------+
```

---

## ⚡ Data Ingestion Flow & Cache Pipeline

To minimize network latencies and prevent API rate-limiting thresholds (under free tiers of financial APIs), the platform uses a **resilient read-through cache mechanism**:

```
[ Request Data for AAPL ]
           |
           v
+------------------------+      YES      +----------------------------+
| Is data cached in DB?  | ------------> | Validate Cache Timestamp   |
+------------------------+               | (within TTL window?)       |
           | NO                          +----------------------------+
           |                                       |
           |                                       | YES
           v                                       v
+------------------------+              [ Return Cached Dataset ]
| Query yfinance / News  |
+------------------------+
           |
           v
+------------------------+
| Write fresh records to |
| SQLite Caching Tables  |
+------------------------+
           |
           v
[ Return Ingested Dataset ]
```

### Multi-Threaded Parallel Scanner Engine
The **Market Scanner** module utilizes a thread-pool executor to scan a configured list of stocks concurrently:
1.  **Worker Pool Allocation:** Allocates concurrent worker threads to retrieve fast market details.
2.  **Vectorized Calculations:** Computes short-term indicators (such as RSI and 1-day change rates) in memory using vectorized pandas operations.
3.  **Result Aggregation:** Safely merges results into a single sorted visual dataframe inside Streamlit within seconds.
