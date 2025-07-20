import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine

# Sets the database host from environment or defaults to localhost
DB_HOST = os.getenv("DB_HOST", "localhost")

# Defines the PostgreSQL database configuration
db_config = {
    "dbname": "sales_conversion",
    "user": "kanikeashritha",
    "password": "ash", 
    "host": DB_HOST,
    "port": "5432"
}

# Defines the path of the CSV file and target table name
csv_path = "Lead Scoring.csv"
table_name = "lead_scoring_data"

# Maps pandas data types to equivalent SQL types
def map_dtype_to_sql(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    else:
        return "TEXT"

# Creates a table if it does not already exist using the dataframe schema
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
    print(f"Table '{table_name}' created (if not exists).")

# Loads the CSV data into PostgreSQL after preparing and cleaning it
def load_csv_to_postgres():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    # Reads CSV and cleans column names and missing values
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.replace({pd.NA: None, pd.NaT: None, "": None})

    # Creates table based on dataframe schema
    create_table_if_not_exists(df, table_name, db_config)

    # Establishes connection and inserts data into the table
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Clears old data from table
    cur.execute(f'DELETE FROM "{table_name}"')

    # Prepares and inserts new records
    placeholders = ','.join(['%s'] * len(df.columns))
    columns = ','.join([f'"{col}"' for col in df.columns])
    insert_query = f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'

    cur.executemany(insert_query, df.values.tolist())
    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {len(df)} rows into '{table_name}'")

# Loads data from PostgreSQL into a pandas dataframe
def load_data_from_postgres(table_name, db_config):
    conn = psycopg2.connect(**db_config)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", con=conn)
    conn.close()
    print("Data loaded successfully from PostgreSQL!")
    return df
