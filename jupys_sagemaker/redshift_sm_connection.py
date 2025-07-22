import psycopg2
import pandas as pd

REDSHIFT_HOST = "wg1.250260913801.ap-south-1.redshift-serverless.amazonaws.com"
REDSHIFT_PORT = 5439
REDSHIFT_DB = "dev"
REDSHIFT_USER = "admin"
REDSHIFT_PASS = "Admin-123"
TABLE_NAME = "leads_dataa"

conn = psycopg2.connect(
    host=REDSHIFT_HOST,
    port=REDSHIFT_PORT,
    dbname=REDSHIFT_DB,
    user=REDSHIFT_USER,
    password=REDSHIFT_PASS,
    sslmode='require'
)

query = f"SELECT * FROM {TABLE_NAME};"
df = pd.read_sql(query, conn)
conn.close()
df.to_csv(f"{TABLE_NAME}.csv", index=False)
print(df.head())
