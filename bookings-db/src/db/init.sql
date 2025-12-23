-- Create the bookings table
CREATE TABLE IF NOT EXISTS bookings (
    id SERIAL PRIMARY KEY,
    hotel_name VARCHAR(255),
    room_id VARCHAR(50),
    room_type VARCHAR(100),
    room_category VARCHAR(100),
    check_in_date DATE,
    check_out_date DATE,
    total_nights INTEGER,
    guest_first_name VARCHAR(100),
    guest_last_name VARCHAR(100),
    guest_email VARCHAR(255),
    guest_phone VARCHAR(50),
    guest_country VARCHAR(100),
    guest_city VARCHAR(100),
    guest_address TEXT,
    guest_zip_code VARCHAR(20),
    meal_plan VARCHAR(50),
    total_price DECIMAL(10, 2)
);