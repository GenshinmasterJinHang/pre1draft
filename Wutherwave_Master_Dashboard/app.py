import os
import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from pymongo import MongoClient

load_dotenv()

#Postgres schema helper
PG_SCHEMA = os.getenv("PG_SCHEMA", "data")   # CHANGE: "public" to your own schema name
def qualify(sql: str) -> str:
    # Replace occurrences of {S}.<table> with <schema>.<table>
    return sql.replace("{S}.", f"{PG_SCHEMA}.")

# CONFIG: Postgres and Mongo Queries
CONFIG = {
    "postgres": {
        "enabled": True,
        "uri": os.getenv("PG_URI", "postgresql+psycopg2://postgres:123456@localhost:5432/music_app"),  # Will read from your .env file
        "queries": {
            
        }
    },

    "mongo": {
        "enabled": True,
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),  # Will read from the .env file
        "db_name": os.getenv("MONGO_DB", "music_app"),               # Will read from the .env file
        
        # CHANGE: Just like above, replace all the following Mongo queries with your own, for the different users you identified
        "queries": {
            
            
        }
    }
}

CONFIG["postgres"]["queries"].update({
    # === User ===
    
    "User: liked tracks (newest first)": {
        "tags": ["end user"],
        "params": ["user_id"],
        "sql": """
        SELECT lt.liked_at, t.track_title, ar.artist_name, al.album_title
        FROM {S}.like_track AS lt
        JOIN {S}.tracks AS t  ON t.track_id = lt.track_id
        JOIN {S}.albums AS al ON al.album_id = t.album_id
        JOIN {S}.artists AS ar ON ar.artist_id = al.artist_id
        WHERE lt.user_id = :user_id
        ORDER BY lt.liked_at DESC;
        """,
        "chart": {"type": "table"}
    },
    "User: my playlists with track counts": {
        "tags": ["end user"],
        "params": ["user_id"],
        "sql": """
        SELECT p.playlist_id, p.name, p.is_public, p.created_at, COUNT(pt.track_id) AS track_count
        FROM {S}.playlists AS p
        LEFT JOIN {S}.playlist_track AS pt ON pt.playlist_id = p.playlist_id
        WHERE p.owner_user_id = :user_id
        GROUP BY p.playlist_id, p.name, p.is_public, p.created_at
        ORDER BY p.created_at DESC;
        """,
        "chart": {"type": "bar", "x": "name", "y": "track_count"}
    },
    "Catalog: new albums in last 90 days": {
        "tags": ["end user"],
        "params": [],
        "sql": """
        SELECT al.album_id, al.album_title, ar.artist_name, al.release_date, COUNT(t.track_id) AS track_count
        FROM {S}.albums AS al
        JOIN {S}.artists AS ar ON ar.artist_id = al.artist_id
        LEFT JOIN {S}.tracks  AS t  ON t.album_id = al.album_id
        WHERE al.release_date >= NOW()::date - INTERVAL '90 days'
        GROUP BY al.album_id, al.album_title, ar.artist_name, al.release_date
        ORDER BY al.release_date DESC;
        """,
        "chart": {"type": "table"}
    },
    "User: new songs from artists I follow (60d)": {
        "tags": ["end user"],
        "params": ["user_id"],
        "sql": """
        SELECT t.track_id, t.track_title, ar.artist_name, al.album_title, al.release_date
        FROM {S}.follow f
        JOIN {S}.artists ar ON ar.artist_id = f.artist_id
        JOIN {S}.albums  al ON al.artist_id = ar.artist_id
        JOIN {S}.tracks  t  ON t.album_id = al.album_id
        WHERE f.follower_user_id = :user_id
          AND al.release_date >= NOW()::date - INTERVAL '60 days'
        ORDER BY al.release_date DESC;
        """,
        "chart": {"type": "table"}
    },
    "User: last 20 listens (with track/artist/album)": {
        "tags": ["end user"],
        "params": ["user_id"],
        "sql": """
        SELECT lh.played_at, t.track_title, ar.artist_name, al.album_title, lh.listened_ms, lh.device_type
        FROM {S}.listen_history AS lh
        JOIN {S}.tracks  AS t  ON t.track_id = lh.track_id
        JOIN {S}.albums  AS al ON al.album_id = t.album_id
        JOIN {S}.artists AS ar ON ar.artist_id = al.artist_id
        WHERE lh.user_id = :user_id
        ORDER BY lh.played_at DESC
        LIMIT 20;
        """,
        "chart": {"type": "table"}
    },

    # === Analysts / Curators ===
    "Trending: top genres by plays (30d)": {
        "tags": ["analyst"],
        "params": [],
        "sql": """
        SELECT g.genre_id, g.genre_name, COUNT(*) AS play_count
        FROM {S}.listen_history AS lh
        JOIN {S}.track_genre AS tg ON tg.track_id = lh.track_id
        JOIN {S}.genres      AS g  ON g.genre_id = tg.genre_id
        WHERE lh.played_at >= NOW() - INTERVAL '30 days'
        GROUP BY g.genre_id, g.genre_name
        ORDER BY play_count DESC
        LIMIT 10;
        """,
        "chart": {"type": "bar", "x": "genre_name", "y": "play_count"}
    },
    "Trending: top 10 artists by unique listeners (30d)": {
        "tags": ["analyst"],
        "params": [],
        "sql": """
        SELECT ar.artist_id, ar.artist_name, COUNT(DISTINCT lh.user_id) AS unique_listeners
        FROM {S}.listen_history AS lh
        JOIN {S}.tracks  AS t  ON t.track_id = lh.track_id
        JOIN {S}.albums  AS al ON al.album_id = t.album_id
        JOIN {S}.artists AS ar ON ar.artist_id = al.artist_id
        WHERE lh.played_at >= NOW() - INTERVAL '30 days'
        GROUP BY ar.artist_id, ar.artist_name
        ORDER BY unique_listeners DESC
        LIMIT 10;
        """,
        "chart": {"type": "bar", "x": "artist_name", "y": "unique_listeners"}
    },
    "Trending: sitewide top 10 tracks by plays (30d)": {
        "tags": ["analyst"],
        "params": [],
        "sql": """
        SELECT t.track_id, t.track_title, ar.artist_name, COUNT(*) AS play_count
        FROM {S}.listen_history AS lh
        JOIN {S}.tracks  AS t  ON t.track_id = lh.track_id
        JOIN {S}.albums  AS al ON al.album_id = t.album_id
        JOIN {S}.artists AS ar ON ar.artist_id = al.artist_id
        WHERE lh.played_at >= NOW() - INTERVAL '30 days'
        GROUP BY t.track_id, t.track_title, ar.artist_name
        ORDER BY play_count DESC
        LIMIT 10;
        """,
        "chart": {"type": "bar", "x": "track_title", "y": "play_count"}
    },
  
    # === Product team ===
    "Product: user subscription status (by user)": {
        "tags": ["product team"],
        "params": ["user_id"],
        "sql": """
        SELECT
            u.user_id,
            u.display_name,
            sp.plan_name,
            sp.monthly_price,
            sp.currency,
            us.start_date,
            us.end_date,
           
            CASE
                WHEN us.start_date <= CURRENT_DATE
                 AND (us.end_date IS NULL OR us.end_date >= CURRENT_DATE)
                THEN TRUE ELSE FALSE
            END AS is_active,
            
            CASE
                WHEN us.end_date IS NOT NULL AND us.end_date >= CURRENT_DATE
                THEN (us.end_date - CURRENT_DATE)
                ELSE NULL
            END AS days_left
        FROM {S}.user_subscription AS us
        JOIN {S}.users              AS u  ON u.user_id = us.user_id
        JOIN {S}.subscription_plan  AS sp ON sp.plan_id = us.plan_id
        WHERE u.user_id = :user_id
        ORDER BY us.start_date DESC, us.end_date DESC NULLS LAST;
        """,
        "chart": {"type": "table"}
    }
})

CONFIG["mongo"]["queries"].update({
    # === User ===
    "Recs: top N for user+scene (latest snapshot)": {
        "tags": ["end user"],
        "params": ["user_id", "scene", "limit"],
        "collection": "user_recs",
        "aggregate": [
            {"$match": {"user_id": "$user_id", "scene": "$scene"}},
            {"$sort": {"generated_at": -1}},
            {"$limit": 1},
            {"$unwind": "$tracks"},
            {"$sort": {"tracks.score": -1}},
            {"$limit": "$limit"},
            {"$project": {
                "_id": 0,
                "generated_at": 1,
                "track_id": "$tracks.track_id",
                "score": "$tracks.score",
                "reasons": "$tracks.reason"
            }}
        ],
        "chart": {"type": "table"}
    },
    "Recs: explain a track (latest snapshot)": {
        "tags": ["end user"],
        "params": ["user_id", "scene", "track_id"],
        "collection": "user_recs",
        "aggregate": [
            {"$match": {"user_id": "$user_id", "scene": "$scene"}},
            {"$sort": {"generated_at": -1}},
            {"$limit": 1},
            {"$unwind": "$tracks"},
            {"$match": {"tracks.track_id": "$track_id"}},
            {"$project": {
                "_id": 0,
                "generated_at": 1,
                "track_id": "$tracks.track_id",
                "score": "$tracks.score",
                "reasons": "$tracks.reason"
            }}
        ],
        "chart": {"type": "table"}
    },
    "Trending tags (7d daily series, by region)": {
        "tags": ["end user", "analyst"],
        "params": ["region"],
        "collection": "trending_tags_daily_ts",
        "aggregate": [
            {"$match": {
                "meta.region": "$region",
                "date": {"$gte": {"$dateSubtract": {"startDate": "$$NOW", "unit": "day", "amount": 7}}}
            }},
            {"$project": {
                "_id": 0,
                "day": {"$dateTrunc": {"date": "$date", "unit": "day"}},
                "tag": "$meta.tag",
                "plays": 1,
                "unique_users": 1
            }},
            {"$sort": {"day": 1, "tag": 1}}
        ],
        "chart": {"type": "table"}
    },

    # === Analyst ===
    "Tags: top 50 by plays (30d, by region)": {
        "tags": ["analyst"],
        "params": [],
        "collection": "trending_tags_daily_ts",
        "aggregate": [
            {"$match": {
                "date": {"$gte": {"$dateSubtract": {"startDate": "$$NOW", "unit": "day", "amount": 30}}}
            }},
            {"$group": {
                "_id": {"tag": "$meta.tag", "region": "$meta.region"},
                "plays": {"$sum": "$plays"},
                "users": {"$sum": "$unique_users"}
            }},
            {"$project": {
                "_id": 0,
                "tag": "$_id.tag",
                "region": "$_id.region",
                "plays": 1,
                "users": 1
            }},
            {"$sort": {"plays": -1}},
            {"$limit": 50}
        ],
        "chart": {"type": "bar", "x": "tag", "y": "plays"}
    },
    "Coverage: users having rec snapshots in last 24h (by scene)": {
        "tags": ["analyst"],
        "params": [],
        "collection": "user_recs",
        "aggregate": [
            {"$match": {
                "generated_at": {"$gte": {"$dateSubtract": {"startDate": "$$NOW", "unit": "hour", "amount": 24}}}
            }},
            {"$group": {
                "_id": "$scene",
                "users": {"$addToSet": "$user_id"}
            }},
            {"$project": {
                "_id": 0,
                "scene": "$_id",
                "user_count": {"$size": "$users"}
            }},
            {"$sort": {"user_count": -1}}
        ],
        "chart": {"type": "bar", "x": "scene", "y": "user_count"}
    },
    "Coverage: track occurrences & avg score (24h)": {
        "tags": ["analyst"],
        "params": [],
        "collection": "user_recs",
        "aggregate": [
            {"$match": {
                "generated_at": {"$gte": {"$dateSubtract": {"startDate": "$$NOW", "unit": "hour", "amount": 24}}}
            }},
            {"$unwind": "$tracks"},
            {"$group": {
                "_id": "$tracks.track_id",
                "count": {"$sum": 1},
                "avg_score": {"$avg": "$tracks.score"}
            }},
            {"$project": {
                "_id": 0,
                "track_id": "$_id",
                "count": 1,
                "avg_score": 1
            }},
            {"$sort": {"count": -1}},
            {"$limit": 100}
        ],
        "chart": {"type": "table"}
    },

    # === System / Product ===
    "SLO: stale latest rec snapshot (>6h) by user+scene": {
        "tags": ["product team"],
        "params": [],
        "collection": "user_recs",
        "aggregate": [
            {"$sort": {"user_id": 1, "scene": 1, "generated_at": -1}},
            {"$group": {
                "_id": {"user_id": "$user_id", "scene": "$scene"},
                "latest_generated_at": {"$first": "$generated_at"}
            }},
            {"$match": {
                "latest_generated_at": {"$lt": {"$dateSubtract": {"startDate": "$$NOW", "unit": "hour", "amount": 6}}}
            }},
            {"$project": {
                "_id": 0,
                "user_id": "$_id.user_id",
                "scene": "$_id.scene",
                "latest_generated_at": 1
            }},
            {"$limit": 100}
        ],
        "chart": {"type": "table"}
    },
    "Guardrail: out-of-range rec scores (expect 0..1)": {
        "tags": ["product team"],
        "params": [],
        "collection": "user_recs",
        "aggregate": [
            {"$unwind": "$tracks"},
            {"$match": {
                "$or": [
                    {"tracks.score": {"$lt": 0}},
                    {"tracks.score": {"$gt": 1}}
                ]
            }},
            {"$project": {
                "_id": 0,
                "user_id": 1,
                "scene": 1,
                "track_id": "$tracks.track_id",
                "score": "$tracks.score",
                "generated_at": 1
            }},
            {"$limit": 100}
        ],
        "chart": {"type": "table"}
    },
    "Ingestion: today's total plays by region": {
        "tags": ["product team"],
        "params": [],
        "collection": "trending_tags_daily_ts",
        "aggregate": [
            {"$match": {
                "date": {"$gte": {"$dateTrunc": {"date": "$$NOW", "unit": "day"}}}
            }},
            {"$group": {
                "_id": "$meta.region",
                "total_plays": {"$sum": "$plays"}
            }},
            {"$project": {
                "_id": 0,
                "region": "$_id",
                "total_plays": 1
            }},
            {"$sort": {"total_plays": 1}}
        ],
        "chart": {"type": "bar", "x": "region", "y": "total_plays"}
    },

    # === NEW: user_similarities for Analyst/Product ===
    "Neighbors: top N similar users for a user": {
        "tags": ["analyst"],
        "params": ["user_id", "limit"],
        "collection": "user_similarities",
        "aggregate": [
            {"$match": {"user_id": "$user_id"}},
            {"$sort": {"updated_at": -1}},
            {"$limit": 1},
            {"$unwind": "$neighbors"},
            {"$sort": {"neighbors.sim": -1}},
            {"$limit": "$limit"},
            {"$project": {
                "_id": 0,
                "neighbor_user_id": "$neighbors.user_id",
                "sim": "$neighbors.sim",
                "updated_at": 1
            }}
        ],
        "chart": {"type": "table"}
    },
    "Neighbors Coverage: neighbor count per user (top 50)": {
        "tags": ["analyst"],
        "params": [],
        "collection": "user_similarities",
        "aggregate": [
            {"$project": {
                "_id": 0,
                "user_id": 1,
                "neighbor_count": {"$size": "$neighbors"},
                "updated_at": 1
            }},
            {"$sort": {"neighbor_count": -1}},
            {"$limit": 50}
        ],
        "chart": {"type": "bar", "x": "user_id", "y": "neighbor_count"}
    },
    "SLO: stale user_similarities (>72h)": {
        "tags": ["product team"],
        "params": [],
        "collection": "user_similarities",
        "aggregate": [
            {"$match": {
                "updated_at": {"$lt": {"$dateSubtract": {"startDate": "$$NOW", "unit": "hour", "amount": 72}}}
            }},
            {"$project": {
                "_id": 0,
                "user_id": 1,
                "updated_at": 1,
                "neighbors_size": {"$size": "$neighbors"}
            }},
            {"$sort": {"updated_at": 1}},
            {"$limit": 100}
        ],
        "chart": {"type": "table"}
    }
})

# The following block of code will create a simple Streamlit dashboard page
st.set_page_config(page_title="Music Platform Analytics Dashboard", layout="wide")
st.title("Music Platform | Analytics Dashboard (Postgres + MongoDB)")

def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for (k, v), c in zip(metrics.items(), cols):
        c.metric(k, v)

@st.cache_resource
def get_pg_engine(uri: str):
    return create_engine(uri, pool_pre_ping=True, future=True)

@st.cache_data(ttl=60)
def run_pg_query(_engine, sql: str, params: dict | None = None):
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

@st.cache_resource
def get_mongo_client(uri: str):
    return MongoClient(uri)

def mongo_overview(client: MongoClient, db_name: str):
    info = client.server_info()
    db = client[db_name]
    colls = db.list_collection_names()
    stats = db.command("dbstats")
    total_docs = sum(db[c].estimated_document_count() for c in colls) if colls else 0
    return {
        "DB": db_name,
        "Collections": f"{len(colls):,}",
        "Total docs (est.)": f"{total_docs:,}",
        "Storage": f"{round(stats.get('storageSize',0)/1024/1024,1)} MB",
        "Version": info.get("version", "unknown")
    }

def replace_mongo_params(stages: list, params: dict) -> list:
    """Replace $param placeholders in MongoDB aggregation stages with actual values."""
    import copy
    stages_copy = copy.deepcopy(stages)
    
    def replace_in_dict(obj):
        if isinstance(obj, dict):
            return {k: replace_in_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_in_dict(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('$') and obj[1:] in params:
            return params[obj[1:]]
        else:
            return obj
    
    return replace_in_dict(stages_copy)

@st.cache_data(ttl=60)
def run_mongo_aggregate(_client, db_name: str, coll: str, stages: list):
    db = _client[db_name]
    docs = list(db[coll].aggregate(stages, allowDiskUse=True))
    return pd.json_normalize(docs) if docs else pd.DataFrame()

def render_chart(df: pd.DataFrame, spec: dict):
    if df.empty:
        st.info("No data found.")
        return
    ctype = spec.get("type", "table")
    # light datetime parsing for x axes
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass

    if ctype == "table":
        st.dataframe(df, use_container_width=True)
    elif ctype == "line":
        st.plotly_chart(px.line(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "bar":
        st.plotly_chart(px.bar(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "pie":
        st.plotly_chart(px.pie(df, names=spec["names"], values=spec["values"]), use_container_width=True)
    elif ctype == "heatmap":
        pivot = pd.pivot_table(df, index=spec["rows"], columns=spec["cols"], values=spec["values"], aggfunc="mean")
        st.plotly_chart(px.imshow(pivot, aspect="auto", origin="upper",
                                  labels=dict(x=spec["cols"], y=spec["rows"], color=spec["values"])),
                        use_container_width=True)
    elif ctype == "treemap":
        st.plotly_chart(px.treemap(df, path=spec["path"], values=spec["values"]), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# The following block of code is for the dashboard sidebar, where you can pick your users, provide parameters, etc.
with st.sidebar:
    st.header("Connections")
    # These fields are pre-filled from .env file
    pg_uri = st.text_input("Postgres URI", CONFIG["postgres"]["uri"])     
    mongo_uri = st.text_input("Mongo URI", CONFIG["mongo"]["uri"])        
    mongo_db = st.text_input("Mongo DB name", CONFIG["mongo"]["db_name"]) 
    st.divider()
    auto_run = st.checkbox("Auto-run on selection change", value=False, key="auto_run_global")

    st.header("Role & Parameters")
    # CHANGE: Change the different roles, the specific attributes, parameters used, etc., to match your own Information System
    role = st.selectbox("User Role", ["end user","analyst","product team"], index=0)
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    track_id = st.number_input("Track ID", min_value=1, value=1, step=1)
    scene = st.selectbox("Scene", ["homepage", "playlist", "radio", "daily_mix", "focus", "commute", "sleep", "party"], index=0)
    region = st.selectbox("Region", [None, "US", "UK", "CA", "DE", "FR", "JP", "KR", "AU", "BR", "IN", "SG", "CN"], index=1)

    PARAMS_CTX = {
        "user_id": int(user_id),
        "track_id": int(track_id),
        "scene": scene,
        "region": region,
        "limit": 20,
    }

#Postgres part of the dashboard
st.subheader("Postgres")
try:
    
    eng = get_pg_engine(pg_uri)

    with st.expander("PostgreSQL Queries", expanded=True):
        # The following will filter queries by role
        def filter_queries_by_role(qdict: dict, role: str) -> dict:
            def ok(tags):
                t = [s.lower() for s in (tags or ["all"])]
                return "all" in t or role.lower() in t
            return {name: q for name, q in qdict.items() if ok(q.get("tags"))}

        pg_all = CONFIG["postgres"]["queries"]
        pg_q = filter_queries_by_role(pg_all, role)

        names = list(pg_q.keys()) or ["(No queries available for this role)"]
        sel = st.selectbox("Select a query", names, key="pg_sel")

        if sel in pg_q:
            q = pg_q[sel]
            sql = qualify(q["sql"])   
            st.code(sql, language="sql")

            run  = auto_run or st.button("▶ Run Query", key="pg_run")
            if run:
                wanted = q.get("params", [])
                params = {k: PARAMS_CTX[k] for k in wanted}
                df = run_pg_query(eng, sql, params=params)
                render_chart(df, q["chart"])
        else:
            st.info("No queries available for this role.")
except Exception as e:
    st.error(f"Postgres error: {e}")

# Mongo panel
if CONFIG["mongo"]["enabled"]:
    st.subheader("🍃 MongoDB")
    try:
        mongo_client = get_mongo_client(mongo_uri)   
        metric_row(mongo_overview(mongo_client, mongo_db))

        with st.expander("MongoDB Aggregation Queries", expanded=True):
            mongo_query_names = list(CONFIG["mongo"]["queries"].keys())
            selm = st.selectbox("Select an aggregation query", mongo_query_names, key="mongo_sel")
            q = CONFIG["mongo"]["queries"][selm]
            st.write(f"**Collection:** `{q['collection']}`")
            st.code(str(q["aggregate"]), language="python")
            runm = auto_run or st.button("▶ Run Aggregation", key="mongo_run")
            if runm:
                wanted = q.get("params", [])
                params = {k: PARAMS_CTX[k] for k in wanted}
                stages = replace_mongo_params(q["aggregate"], params)
                dfm = run_mongo_aggregate(mongo_client, mongo_db, q["collection"], stages)
                render_chart(dfm, q["chart"])
    except Exception as e:
        st.error(f"Mongo error: {e}")