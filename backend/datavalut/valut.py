import datetime
import os
import string
import psycopg2
import json
import google.generativeai as genai
import re
from psycopg2 import sql
from dotenv import load_dotenv
import uuid
import secrets

class DataValut:
    """
    A class to create a filtered, temporary clone of a source PostgreSQL database
    based on a natural language prompt, using a GenAI model to select relevant data
    and a hardcoded filter to ensure sensitive data is removed.
    """
    DB_LOG_FILE = "scheduled_dbs.json"

    def __init__(self, src_db_str: str, progress_callback=None):
        """
        Initializes the DBFilter instance.

        Args:
            src_db_str (str): The connection string for the source database.
                              Example: "postgresql://user:pass@host:port/dbname"
            progress_callback (callable): Optional callback function for progress updates
        """
        load_dotenv()

        # 🔐 Configure Gemini
        try:
            genai.configure(api_key=os.getenv("GENAI_API_KEY"))
            self.model = genai.GenerativeModel("gemini-2.5-pro")
        except Exception as e:
            raise ValueError(f"Failed to configure Gemini. Ensure GENAI_API_KEY is set in your .env file. Error: {e}")

        # 🌍 Source DB info
        self.SRC_DB_STR = src_db_str
        self.SRC_DB = psycopg2.extensions.parse_dsn(self.SRC_DB_STR)
        
        # Progress callback for real-time updates
        self.progress_callback = progress_callback

        # Superuser connection string to create the target DB
        superuser_db_str = os.getenv("SUPERUSER_DB_STR")
        if not superuser_db_str:
            raise ValueError("SUPERUSER_DB_STR is not set in the .env file.")
        self.SUPERUSER_DB = psycopg2.extensions.parse_dsn(superuser_db_str)
        
        # 🛡️ Hardcoded security layer: patterns for sensitive columns to exclude.
        # This list is checked AFTER the AI gives its response.
        self.SENSITIVE_COL_PATTERNS = [
            'password', 'token', 'secret', 'key', 'auth', 'credit', 'card', 
            'cvv', 'cvc', 'ssn', 'social_security'
        ]
        
        print("✅ DBFilter initialized successfully.")
        if self.progress_callback:
            self.progress_callback(0.05, "DataValut initialized", "DataValut instance created successfully")

    def _send_progress(self, progress: float, step: str, message: str):
        """Helper method to send progress updates if callback is available"""
        if self.progress_callback:
            self.progress_callback(progress, step, message)
        print(f"[{progress*100:.1f}%] {step}: {message}")

    def _create_target_database(self) -> dict:
        # ... (This function remains the same as before)
        self._send_progress(0.1, "Creating database", "Generating random database credentials...")
        
        random_username = "user_" + uuid.uuid4().hex[:8]
        alphabet = string.ascii_letters + string.digits + ".#_-"
        random_password = ''.join(secrets.choice(alphabet) for _ in range(16))
        random_dbname = "db_" + uuid.uuid4().hex[:8]

        self._send_progress(0.15, "Connecting to database", "Establishing superuser connection...")
        conn = psycopg2.connect(**self.SUPERUSER_DB)
        conn.autocommit = True
        cur = conn.cursor()

        try:
            self._send_progress(0.2, "Creating user", f"Creating database user: {random_username}")
            cur.execute(sql.SQL("CREATE USER {} WITH PASSWORD %s").format(sql.Identifier(random_username)), [random_password])
            print(f"✅ Created user: {random_username}")

            self._send_progress(0.25, "Creating database", f"Creating database: {random_dbname}")
            cur.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(random_dbname),
                sql.Identifier(random_username)
            ))
            print(f"✅ Created DB: {random_dbname}")
        finally:
            cur.close()
            conn.close()

        self._send_progress(0.3, "Logging database", "Recording database creation details...")
        # NOTE: For production, consider not logging the password.
        db_record = {
            "username": random_username,
            "password": random_password, 
            "dbname": random_dbname,
            "created_at": datetime.datetime.utcnow().isoformat()
        }

        data = []
        if os.path.exists(self.DB_LOG_FILE):
            with open(self.DB_LOG_FILE, "r") as f:
                try: data = json.load(f)
                except json.JSONDecodeError: pass
        
        data.append(db_record)
        with open(self.DB_LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)

        return {
            "dbname": random_dbname, "user": random_username, "password": random_password,
            "host": self.SUPERUSER_DB["host"], "port": self.SUPERUSER_DB["port"]
        }

    def _get_full_schema(self, conn) -> dict:
        # ... (This function remains the same as before)
        self._send_progress(0.35, "Reading schema", "Analyzing source database structure...")
        
        cur = conn.cursor()
        cur.execute("""
            SELECT table_schema, table_name FROM information_schema.tables
            WHERE table_type = 'BASE TABLE' AND table_schema NOT IN ('information_schema', 'pg_catalog');
        """)
        rows = cur.fetchall()

        if not rows: 
            print("⚠️ No tables found.")
            self._send_progress(0.4, "Schema analysis", "No tables found in source database")
        else: 
            print(f"✅ Tables found: {[f'{s}.{t}' for s, t in rows]}")
            self._send_progress(0.4, "Schema analysis", f"Found {len(rows)} tables in source database")

        schema = {}
        total_tables = len(rows)
        for i, (schema_name, table_name) in enumerate(rows):
            progress = 0.4 + (i / total_tables) * 0.1  # Progress from 0.4 to 0.5
            self._send_progress(progress, "Reading columns", f"Analyzing table: {table_name}")
            
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s;
            """, (schema_name, table_name))
            columns = cur.fetchall()
            schema[table_name] = [{"name": col[0], "type": col[1]} for col in columns]

        cur.close()
        self._send_progress(0.5, "Schema complete", f"Schema analysis complete for {len(schema)} tables")
        return schema

    def _filter_schema_with_gemini(self, full_schema: dict, prompt: str) -> dict:
        # ... (This function remains the same as before, with the improved prompt)
        self._send_progress(0.55, "AI Analysis", "Preparing request for Gemini AI...")
        
        chat_prompt = f"""
Given the database schema below and the task: "{prompt}", return only the necessary tables and columns in JSON format.

RULES:
1. Your entire response must be ONLY the JSON object.
2. Do NOT include any other text, explanations, or markdown formatting like ```json.
3. The JSON keys must be the table names, and the values must be an array of column name strings.
4. Exclude any columns that are considered sensitive PII (Personally Identifiable Information), such as passwords, api_keys, tokens, credit card numbers, etc.

Full Database Schema:
{json.dumps(full_schema, indent=2)}
"""
        try:
            print("🤖 Sending request to Gemini API...")
            self._send_progress(0.6, "AI Processing", "Sending schema to Gemini AI for analysis...")
            
            response = self.model.generate_content(chat_prompt)
            
            self._send_progress(0.7, "AI Response", "Processing Gemini AI response...")
            res_clean = re.sub(r"```(json)?", "", response.text, flags=re.MULTILINE).strip()
            filtered_schema = json.loads(res_clean)
            if not isinstance(filtered_schema, dict):
                print(f"❌ Gemini returned valid JSON, but not a dictionary (type: {type(filtered_schema)}).")
                self._send_progress(0.7, "AI Error", "Gemini returned invalid response format")
                return {}
            
            self._send_progress(0.75, "AI Complete", f"Gemini successfully analyzed schema - {len(filtered_schema)} tables selected")
            return filtered_schema
        except json.JSONDecodeError as e:
            print(f"❌ Gemini response failed JSON parsing. Error: {e}")
            print(f"--- Raw Gemini Response ---\n{response.text}\n-------------------------")
            self._send_progress(0.7, "AI Error", f"Failed to parse Gemini response: {str(e)}")
            return {}
        except Exception as e:
            print(f"❌ A critical error occurred while calling the Gemini API: {e}")
            self._send_progress(0.7, "AI Error", f"Gemini API error: {str(e)}")
            return {}

    def _sanitize_schema(self, schema_from_ai: dict) -> dict:
        """
        Applies a hardcoded denylist to the schema provided by the AI.
        This is a critical security step to ensure sensitive columns are never included.
        """
        print("\n🛡️ Applying security sanitization filter...")
        self._send_progress(0.8, "Security Filter", "Applying security sanitization to AI-selected schema...")
        
        sanitized_schema = {}
        removed_columns = 0
        
        for table, columns in schema_from_ai.items():
            safe_columns = []
            for col in columns:
                # Check if any sensitive pattern is a substring of the column name (case-insensitive)
                if any(pattern in col.lower() for pattern in self.SENSITIVE_COL_PATTERNS):
                    print(f"  - Removing sensitive column '{col}' from table '{table}'.")
                    removed_columns += 1
                else:
                    safe_columns.append(col)
            
            # Only include the table if it still has columns after sanitization
            if safe_columns:
                sanitized_schema[table] = safe_columns

        self._send_progress(0.85, "Security Complete", f"Security filter complete - removed {removed_columns} sensitive columns")
        return sanitized_schema
        
    def _reset_db(self, conn):
        # ... (This function remains the same)
        cur = conn.cursor()
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        conn.commit()
        cur.close()

    def _create_target_schema(self, tgt_conn, filtered_schema: dict, full_schema: dict):
        # ... (This function remains the same)
        self._send_progress(0.87, "Creating Schema", "Creating tables in target database...")
        
        cur = tgt_conn.cursor()
        created_tables = 0
        
        for table, cols in filtered_schema.items():
            if table not in full_schema: continue
            col_defs = [f'"{col_info["name"]}" {col_info["type"]}' for col_info in full_schema[table] if col_info["name"] in cols]
            if col_defs:
                create_table_sql = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({});").format(sql.Identifier(table), sql.SQL(", ".join(col_defs)))
                cur.execute(create_table_sql)
                created_tables += 1
                
        tgt_conn.commit()
        cur.close()
        self._send_progress(0.9, "Schema Created", f"Created {created_tables} tables in target database")

    def _transfer_data(self, src_conn, tgt_conn, filtered_schema: dict):
        # ... (This function remains the same)
        self._send_progress(0.92, "Transferring Data", "Starting data transfer to target database...")
        
        src_cur = src_conn.cursor()
        tgt_cur = tgt_conn.cursor()
        total_rows = 0
        total_tables = len(filtered_schema)
        
        for i, (table, columns) in enumerate(filtered_schema.items()):
            if not columns: continue
            
            progress = 0.92 + (i / total_tables) * 0.07  # Progress from 0.92 to 0.99
            self._send_progress(progress, "Data Transfer", f"Transferring data for table: {table}")
            
            col_ids = [sql.Identifier(c) for c in columns]
            placeholders = sql.SQL(',').join(sql.Placeholder() * len(columns))
            select_sql = sql.SQL("SELECT {} FROM {}").format(sql.SQL(', ').join(col_ids), sql.Identifier(table))
            insert_sql = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(sql.Identifier(table), sql.SQL(', ').join(col_ids), placeholders)
            
            src_cur.execute(select_sql)
            rows = src_cur.fetchall()
            if rows:
                tgt_cur.executemany(insert_sql, rows)
                total_rows += len(rows)
                print(f"  -> Transferred {len(rows)} rows to table '{table}'.")
                
        tgt_conn.commit()
        src_cur.close()
        tgt_cur.close()
        self._send_progress(0.99, "Transfer Complete", f"Data transfer complete - {total_rows} total rows transferred")

    def create_filtered_db(self, prompt: str) -> dict:
        """
        Main orchestration method. Creates and populates a new database based on the prompt.
        """
        print(f"\n🚀 Starting filtered database creation for prompt: '{prompt}'")
        try:
            TGT_DB = self._create_target_database()
            print(f"✅ Target database placeholder created: {TGT_DB['dbname']}")
        except Exception as e:
            print(f"❌ Critical Error: Could not create target database. {e}")
            return {}

        try:
            with psycopg2.connect(**self.SRC_DB) as src_conn, psycopg2.connect(**TGT_DB) as tgt_conn:
                # Step 1: Get full schema
                print("\n📦 Dumping full schema from source DB...")
                full_schema = self._get_full_schema(src_conn)
                if not full_schema: return {}

                # Step 2: Ask Gemini to select a schema subset
                gemini_schema = self._filter_schema_with_gemini(full_schema, prompt)
                if not gemini_schema: return {}
                print(f"📤 Gemini suggested schema: {json.dumps(gemini_schema, indent=2)}")

                # Step 3: Sanitize the schema from Gemini (CRITICAL SECURITY STEP)
                filtered_schema = self._sanitize_schema(gemini_schema)
                if not filtered_schema:
                    print("❌ Halting: After sanitization, no data was left to copy.")
                    return {}
                print(f"✅ Final sanitized schema: {json.dumps(filtered_schema, indent=2)}")

                # Step 4: Create schema and transfer data
                print("\n🏗️ Resetting and creating new schema in target DB...")
                self._reset_db(tgt_conn)
                self._create_target_schema(tgt_conn, filtered_schema, full_schema)
                
                print("\n📤 Copying selected data to target DB...")
                self._transfer_data(src_conn, tgt_conn, filtered_schema)
                
                print("\n\n✅ Data transfer complete.")
                print("--- Transfer Report ---")
                conn_str = f"postgresql://{TGT_DB['user']}:{TGT_DB['password']}@{TGT_DB['host']}:{TGT_DB['port']}/{TGT_DB['dbname']}"
                print(f"Connection String: {conn_str}")
                
                self._send_progress(1.0, "Complete", f"Database creation complete: {TGT_DB['dbname']}")
                return TGT_DB

        except psycopg2.Error as e:
            print(f"\n❌ A database error occurred: {e}")
            return {}
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            return {}


if __name__ == "__main__":
    # --- EXAMPLE USAGE ---
    # 1. Create a `.env` file in the same directory with these lines:
    # GENAI_API_KEY="your_google_ai_api_key"
    # SUPERUSER_DB_STR="postgresql://postgres:your_superuser_password@localhost:5432/postgres"

    # 2. Set your source database connection string here.
    #    This example assumes a local DB named 'full_db' with user 'test_user'.
    SRC_DB_CONNECTION_STRING = "postgresql://test_user:12345678@localhost:5432/full_db"
    
    try:
        db_filter = DataValut(src_db_str=SRC_DB_CONNECTION_STRING)

        user_prompt = "I need to analyze user sign-up trends. I need user names, emails, and their creation dates, but I absolutely cannot have any passwords or tokens."
        
        new_db_details = db_filter.create_filtered_db(prompt=user_prompt)

        if new_db_details:
            print("\n--- ✅ Process Complete ---")
            print(f"Successfully created new filtered database: {new_db_details['dbname']}")
        else:
            print("\n--- ❌ Process Failed ---")
            print("Could not create the filtered database. Please check logs.")
            
    except (ValueError, psycopg2.OperationalError) as e:
        print(f"Initialization or Connection Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the main block: {e}")