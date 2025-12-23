#!/bin/bash
# Script to initialize the database and load data

# Wait for PostgreSQL to start
echo "Waiting for PostgreSQL to start..."
until PGPASSWORD=$POSTGRES_PASSWORD psql -h bookings-db -U $POSTGRES_USER -c '\q'; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done
echo "PostgreSQL is up - executing command"

# Check if database exists
DB_EXISTS=$(PGPASSWORD=$POSTGRES_PASSWORD psql -h bookings-db -U $POSTGRES_USER -t -c "SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB';")
if [ -z "$DB_EXISTS" ]; then
    echo "Database '$POSTGRES_DB' does not exist. Creating it..."
    PGPASSWORD=$POSTGRES_PASSWORD psql -h bookings-db -U $POSTGRES_USER -c "CREATE DATABASE $POSTGRES_DB;"
else
    echo "Database '$POSTGRES_DB' already exists."
fi

# Check if table exists
if ! PGPASSWORD=$POSTGRES_PASSWORD psql -h bookings-db -U $POSTGRES_USER -d $POSTGRES_DB -c "\dt" | grep -qw bookings; then
    echo "Table 'bookings' does not exist. Creating it..."
    PGPASSWORD=$POSTGRES_PASSWORD psql -h bookings-db -U $POSTGRES_USER -d $POSTGRES_DB -f /app/db/init.sql
else
    echo "Table 'bookings' already exists."
fi

# Check if table is empty
if [ "$(PGPASSWORD=$POSTGRES_PASSWORD psql -h bookings-db -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT COUNT(*) FROM bookings;")" -eq 0 ]; then
    echo "Table 'bookings' is empty. Loading data..."
    POSTGRES_USER=$POSTGRES_USER POSTGRES_PASSWORD=$POSTGRES_PASSWORD POSTGRES_DB=$POSTGRES_DB python /app/db/load_data.py
else
    echo "Table 'bookings' already contains data. Skipping data load."
fi

# Configure logging if enabled
if [ "${DATABASE_CONFIG_LOGGING:-NO}" = "YES" ]; then
    echo "Configuring PostgreSQL logging..."
    PGPASSWORD=$POSTGRES_PASSWORD psql -h bookings-db -U $POSTGRES_USER -d $POSTGRES_DB << EOF
    ALTER SYSTEM SET log_statement = 'all';
    ALTER SYSTEM SET log_duration = 'on';
    ALTER SYSTEM SET log_min_duration_statement = 0;
    SELECT pg_reload_conf();
EOF
    echo "PostgreSQL logging configuration completed."
else
    echo "Logging configuration is disabled."
fi

echo "Database initialization completed."
exit 0 