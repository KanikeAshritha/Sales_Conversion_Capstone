import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine

DB_HOST = os.getenv("DB_HOST", "localhost")

# PostgreSQL connection config
db_config = {
    "dbname": "sales_conversion",
    "user": "kanikeashritha",
    "password": "ash",  # Keep this in .env or secure vault in real project
    "host": DB_HOST,
    "port": "5432"
}

# CSV and table setup
csv_path = os.path.join("data", "raw", "Lead Scoring.csv")
table_name = "lead_scoring_data"


# ✅ Map pandas dtype to SQL types
def map_dtype_to_sql(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    else:
        return "TEXT"


# ✅ Create table dynamically from dataframe schema
def create_table_if_not_exists(df, table_name, db_config):
    cols_with_types = [
        f'"{col}" {map_dtype_to_sql(dtype)}'
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    create_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(cols_with_types)}
        );
    """

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute(create_query)
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Table '{table_name}' created (if not exists).")


# ✅ Load CSV to PostgreSQL table
def load_csv_to_postgres():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.replace({pd.NA: None, pd.NaT: None, "": None})

    # Create table first
    create_table_if_not_exists(df, table_name, db_config)

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute(f'DELETE FROM "{table_name}"')  # Clean old data

    placeholders = ','.join(['%s'] * len(df.columns))
    columns = ','.join([f'"{col}"' for col in df.columns])
    insert_query = f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'

    cur.executemany(insert_query, df.values.tolist())
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Inserted {len(df)} rows into '{table_name}'")


# ✅ Read table into pandas DataFrame
def load_data_from_postgres(table_name, db_config):
    from sqlalchemy import create_engine
    import pandas as pd

    try:
        engine_str = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        )
        engine = create_engine(engine_str)

        # Read table
        df = pd.read_sql_table(table_name, con=engine)

        # Convert all column names to string
        df.columns = [str(col) for col in df.columns]

        print(f"📦 Loaded {len(df)} rows from '{table_name}'")
        return df

    except Exception as e:
        print(f"❌ Error loading data from PostgreSQL: {e}")
        return pd.DataFrame()

