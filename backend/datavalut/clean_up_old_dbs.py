# cleanup_old_dbs.py

import os
import json
import datetime
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()

SUPERUSER_DB = psycopg2.extensions.parse_dsn(os.getenv("SUPERUSER_DB_STR"))
DB_LOG_FILE = "scheduled_dbs.json"
TTL_MINUTES = 30  # time to live

def cleanup_expired_dbs():
    if not os.path.exists(DB_LOG_FILE):
        return

    with open(DB_LOG_FILE, "r") as f:
        entries = json.load(f)

    keep = []
    now = datetime.datetime.utcnow()
    conn = psycopg2.connect(**SUPERUSER_DB)
    conn.autocommit = True
    cur = conn.cursor()

    for entry in entries:
        created = datetime.datetime.fromisoformat(entry["created_at"])
        age = (now - created).total_seconds() / 60.0
        if age >= TTL_MINUTES:
            print(f"🗑 Deleting DB: {entry['dbname']}, User: {entry['username']} (Age: {int(age)} min)")
            try:
                cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(entry["dbname"])))
                cur.execute(sql.SQL("DROP USER IF EXISTS {}").format(sql.Identifier(entry["username"])))
            except Exception as e:
                print(f"❌ Error while deleting {entry['dbname']}: {e}")
        else:
            keep.append(entry)

    with open(DB_LOG_FILE, "w") as f:
        json.dump(keep, f, indent=2)

    cur.close()
    conn.close()

if __name__ == "__main__":
    cleanup_expired_dbs()
