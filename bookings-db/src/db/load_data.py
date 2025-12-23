#!/usr/bin/env python3
"""Script to load booking data from Excel into PostgreSQL database."""

import os
import pandas as pd
import psycopg2
from psycopg2 import OperationalError, DatabaseError

def check_table_exists(cursor, table_name):
    """Check if a table exists in the database."""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        );
    """, (table_name,))
    return cursor.fetchone()[0]

def execute_sql_file(cursor, file_path):
    """Execute SQL commands from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        sql_commands = file.read()
        cursor.execute(sql_commands)

def load_excel_to_postgres():
    """Load booking data from Excel file into PostgreSQL database."""
    try:
        # Connect to PostgreSQL using environment variables
        conn = psycopg2.connect(
            host="bookings-db",
            database=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD')
        )

        # Create a cursor
        cursor = conn.cursor()

        # Check if table exists and create it if it doesn't
        if not check_table_exists(cursor, 'bookings'):
            print("Table 'bookings' does not exist. Creating it...")
            execute_sql_file(cursor, '/app/db/init.sql')
            conn.commit()

        # Read the Excel file
        excel_file = "/app/data/all_bookings.xlsx"
        df = pd.read_excel(excel_file)

        # Convert date columns to datetime
        df['Check-in Date'] = pd.to_datetime(df['Check-in Date'])
        df['Check-out Date'] = pd.to_datetime(df['Check-out Date'])
        
        # Calculate total nights
        df['Total Nights'] = (df['Check-out Date'] - df['Check-in Date']).dt.days

        # Insert data into the database
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO bookings (
                    hotel_name, room_id, room_type, room_category,
                    check_in_date, check_out_date, total_nights, guest_first_name,
                    guest_last_name, guest_email, guest_phone,
                    guest_country, guest_city, guest_address,
                    guest_zip_code, meal_plan, total_price
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row['Hotel Name'], row['Room ID'], row['Room Type'],
                row['Room Category'], row['Check-in Date'], row['Check-out Date'],
                row['Total Nights'], row['Guest First Name'], row['Guest Last Name'], row['Guest Email'],
                row['Guest Phone'], row['Guest Country'], row['Guest City'],
                row['Guest Address'], row['Guest Zip Code'], row['Meal Plan'],
                row['Total Price']
            ))

        # Commit the transaction
        conn.commit()
        print("Data loaded successfully into the database.")

    except (OperationalError, DatabaseError, FileNotFoundError, ValueError) as error:
        print(f"Error while connecting to PostgreSQL: {error}")
    finally:
        if conn:
            cursor.close()
            conn.close()
            print("PostgreSQL connection is closed.")

if __name__ == "__main__":
    load_excel_to_postgres()
