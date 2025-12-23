import os
from langchain_community.utilities import SQLDatabase

def get_database():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "bookings-db")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")

    database_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    return SQLDatabase.from_uri(database_uri)


# Check db conn and print sample data
#db_global = get_database()
#print("Database connection established.")
#
## Probar ejecución de consulta simple
#try:
#    result = db_global.run("SELECT 1;")
#    tables = db_global.run("""
#SELECT table_name
#FROM information_schema.tables
#WHERE table_schema = 'public';
#""")
#    rows = db_global.run("SELECT * FROM bookings LIMIT 10;")
#    print("Sample rows from bookings:")
#    for row in rows:
#        print(row)
#    print("✅ Connection test successful. Query result:", result)
#except Exception as e:
#    print("❌ Connection test failed:", e)