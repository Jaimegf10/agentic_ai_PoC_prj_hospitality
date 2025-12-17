"""Module for generating synthetic hotel and booking data."""

import os
import sys
import time
import yaml

# Add parent directory to path to allow running directly: python gen_synthetic_hotels.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator.hotel_generator import generate_hotels
from src.generator.booking_generator import generate_hotel_bookings
from src.generator.hotel_query_generator import HotelQueryGenerator
from src.output.booking_output_writer import \
    generate_file_md_hotel_bookings, \
    generate_file_excel_all_bookings
from src.output.hotel_output_writer import \
    generate_file_json_for_hotels, \
    generate_file_excel_for_hotels, \
    generate_file_csv_for_hotels, \
    generate_file_csv_for_all_hotels, \
    generate_file_md_hotel_details, \
    generate_file_md_hotel_rooms
from src.output.hotel_query_writer import generate_file_csv_for_queries_room_hotels


def load_config(config_path="../config/generate_hotels_param.yaml"):
    """Load the configuration file for hotel generation.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The loaded configuration.
    """
    script_dir = os.path.dirname(__file__)
    abs_config_path = os.path.join(script_dir, config_path)
    with open(abs_config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_config_queries(config_path="../config/hotel_queries.yaml"):
    """Load the configuration file for hotel queries.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The loaded configuration. 
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file '{config_path}' does not exist.")

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        raise RuntimeError(f"Failed to read the configuration file '{config_path}': {e}") from e

    room_queries = config.get('room_queries', {})
    hotel_room_queries = room_queries.get('hotel', [])
    hotel_compare_queries = room_queries.get('hotel_compare', [])
    organization_room_queries = room_queries.get('organization', [])
    num_queries = room_queries.get('number', None)

    if not hotel_room_queries:
        raise ValueError("The list of hotel room queries is empty in the configuration file.")
    if not hotel_compare_queries:
        raise ValueError("The list of hotel compare queries is empty in the configuration file.")
    if not organization_room_queries:
        raise ValueError(
            "The list of organization room queries is empty in the configuration file."
        )
    if num_queries is None:
        raise ValueError("The number of queries is not specified in the configuration file.")

    return config

if __name__ == "__main__":
    start_time = time.time()
    # Get absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # bookings-db/
    
    print(f"script_dir: {script_dir}")
    print(f"project_root: {project_root}")
    
    hotelGenerationConfig = load_config(os.path.join(script_dir,"../config/generate_hotels_param.yaml"))
    queries_config = load_config_queries(os.path.join(script_dir,"../config/hotel_queries.yaml"))
    print(f"hotelGenerationConfig: {os.path.join(script_dir,'../config/generate_hotels_param.yaml')}")
    print(f"queries_config: {os.path.join(script_dir, '../config/hotel_queries.yaml')}")
    
    
    # Resolve output paths from config (relative paths are resolved from project_root)
    config_hotels_path = hotelGenerationConfig["process"]["output_path_hotels"]
    config_bookings_path = hotelGenerationConfig["process"]["output_path_bookings"]
    
    OUTPUT_PATH_HOTELS = config_hotels_path if os.path.isabs(config_hotels_path) else os.path.join(project_root, config_hotels_path)
    OUTPUT_PATH_BOOKINGS = config_bookings_path if os.path.isabs(config_bookings_path) else os.path.join(project_root, config_bookings_path)
    
    print(f"OUTPUT_PATH_HOTELS: {OUTPUT_PATH_HOTELS}")
    print(f"OUTPUT_PATH_BOOKINGS: {OUTPUT_PATH_BOOKINGS}")
    
    os.makedirs(OUTPUT_PATH_HOTELS, exist_ok=True)
    os.makedirs(OUTPUT_PATH_BOOKINGS, exist_ok=True)
    hotel_list = generate_hotels(hotelGenerationConfig)
    generate_file_excel_for_hotels(hotel_list,OUTPUT_PATH_HOTELS)
    generate_file_json_for_hotels(hotel_list,OUTPUT_PATH_HOTELS)
    generate_file_csv_for_hotels(hotel_list,OUTPUT_PATH_HOTELS)
    generate_file_csv_for_all_hotels(hotel_list, OUTPUT_PATH_HOTELS)
    # Generate queries using the generator
    query_generator = HotelQueryGenerator(queries_config)
    hotel_names = [hotel["Name"] for hotel in hotel_list]
    queries = query_generator.get_room_queries(hotel_names)
    generate_file_csv_for_queries_room_hotels(queries, OUTPUT_PATH_HOTELS)
    generate_file_md_hotel_details(hotel_list, OUTPUT_PATH_HOTELS)
    generate_file_md_hotel_rooms(hotel_list, OUTPUT_PATH_HOTELS)


    hotel_booking_list = []

    for hotel_to_book in hotel_list:
        hotel_bookings = generate_hotel_bookings(hotel_to_book, hotelGenerationConfig)
        # generate_file_json_for_bookings(hotel_bookings, hotel_to_book["hotelkey"],
        #       hotel_to_book["Name"], OUTPUT_PATH_BOOKINGS)
        # generate_file_excel_for_bookings(hotel_bookings, hotel_to_book["hotelkey"],
        #           hotel_to_book["Name"], OUTPUT_PATH_BOOKINGS)
        hotel_booking_list.append(hotel_bookings)

    generate_file_md_hotel_bookings(hotel_booking_list, OUTPUT_PATH_HOTELS)
    generate_file_excel_all_bookings(hotel_booking_list,os.path.join(OUTPUT_PATH_BOOKINGS, "all_bookings.xlsx"))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Synthetic Data successfully generated {elapsed_time:.2f} sg")
