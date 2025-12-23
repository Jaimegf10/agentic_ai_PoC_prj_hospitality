"""Output package for writing hotel and booking data to various file formats."""

from .booking_output_writer import (
    generate_file_json_for_bookings,
    generate_file_excel_for_bookings,
    generate_file_md_hotel_bookings,
    generate_file_excel_all_bookings
)
from .hotel_output_writer import (
    generate_file_json_for_hotels,
    generate_file_excel_for_hotels,
    generate_file_csv_for_hotels,
    generate_file_csv_for_all_hotels,
    generate_file_md_hotel_details,
    generate_file_md_hotel_rooms
)
from .hotel_query_writer import generate_file_csv_for_queries_room_hotels

__all__ = [
    # Booking output functions
    'generate_file_json_for_bookings',
    'generate_file_excel_for_bookings',
    'generate_file_md_hotel_bookings',
    'generate_file_excel_all_bookings',

    # Hotel output functions
    'generate_file_json_for_hotels',
    'generate_file_excel_for_hotels',
    'generate_file_csv_for_hotels',
    'generate_file_csv_for_all_hotels',
    'generate_file_md_hotel_details',
    'generate_file_md_hotel_rooms',

    # Query output functions
    'generate_file_csv_for_queries_room_hotels'
]
