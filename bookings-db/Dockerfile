FROM python:3.12-slim-bookworm

# Install system dependencies and security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir pandas openpyxl psycopg2-binary

# Create app directory
WORKDIR /app

# Copy the application code
COPY src/ /app/

# Create data directory
RUN mkdir -p /app/data

# Copy the Excel file
COPY output_files/bookings/all_bookings.xlsx /app/data/

# Make the initialization script executable
RUN chmod +x /app/db/init-db.sh

# Expose PostgreSQL port
EXPOSE 5432

# Start PostgreSQL and initialize the database
CMD ["/app/db/init-db.sh"]