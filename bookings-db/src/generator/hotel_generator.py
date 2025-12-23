"""Module for generating synthetic hotel data including rooms and parameters."""

import random
import re
import os
from . import parametric_utils as ParUt
from . import hotel_name_location_generator

def generate_parametrization(config):
    """
    Generate hotel parametrization based on configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: Hotel parametrization parameters
    """
    return {
        "OccupancyPeakSeasonWeight": random.randint(
            config["hotel_occupancy"]["occupancy_weight"]["peak_season"]["min"],
            config["hotel_occupancy"]["occupancy_weight"]["peak_season"]["max"]
        ),
        "OccupancyOffSeasonWeight": random.randint(
            config["hotel_occupancy"]["occupancy_weight"]["off_season"]["min"],
            config["hotel_occupancy"]["occupancy_weight"]["off_season"]["max"]
        ),
        "OccupancyBaseDiscountPercentage": random.randint(
            config["pricing"]["reduce_hosts_price_discount_percentage"]["min"],
            config["pricing"]["reduce_hosts_price_discount_percentage"]["max"]
        ),
        "ExtraBedChargePercentage": random.randint(
            config["pricing"]["extra_bed_price_increase_percentage"]["min"],
            config["pricing"]["extra_bed_price_increase_percentage"]["max"]
        ),
        "MealPlanWeights": ParUt.get_hotel_mealplan_weight(config['hotel_occupancy']),
        "MealPlanPrices": ParUt.get_meal_plan_prices(
            config['hotel_occupancy']["meal_plans_weight"]
        ),
        "PromotionPriceDiscount": random.randint(
            config["pricing"]["promotion_price_discount_percentage"]["min"],
            config["pricing"]["promotion_price_discount_percentage"]["max"]
        )
    }

def generate_rooms(config):
    """
    Generate room configurations for a hotel.

    Args:
        config (dict): Configuration dictionary

    Returns:
        list: List of room configurations
    """
    # Get basic hotel parameters
    hotel_params = {
        'num_rooms': ParUt.get_rooms_floors(config["rooms_per_hotel"])[0],
        'num_floors': ParUt.get_rooms_floors(config["rooms_per_hotel"])[1],
        'room_type_weights': ParUt.get_room_type_weights(config["rooms_per_hotel"]),
        'room_category_premium_weight': ParUt.get_room_category_premium_weight(
            config["rooms_per_hotel"]
        ),
        'base_prices': ParUt.get_standard_low_season_prices(config["pricing"]),
        'premium_increase': ParUt.get_premium_increase(config["pricing"]),
        'high_season_increase': ParUt.get_high_season_increase(config["pricing"])
    }

    # Calculate room distribution
    hotel_params['rooms_per_floor'] = hotel_params['num_rooms'] // hotel_params['num_floors']
    hotel_params['extra_rooms'] = hotel_params['num_rooms'] % hotel_params['num_floors']

    rooms = []
    for floor in range(1, hotel_params['num_floors'] + 1):
        # Calculate rooms for this floor
        extra_room = 1 if floor <= hotel_params['extra_rooms'] else 0
        num_rooms_on_floor = hotel_params['rooms_per_floor'] + extra_room

        for room_number in range(1, num_rooms_on_floor + 1):
            room_id = f"{str(floor).zfill(2)}-{str(room_number).zfill(3)}"
            category = ParUt.get_room_category(hotel_params['room_category_premium_weight'])
            room_guests = ParUt.get_room_guests(hotel_params['room_type_weights'])
            room_type = ParUt.get_room_type_name(room_guests)
            base_price = round(float(hotel_params['base_prices'][room_guests]), 2)
            base_price = ParUt.get_category_price(category,
                                                base_price,
                                                hotel_params['premium_increase'])
            high_season_price = round(
                base_price * (hotel_params['high_season_increase'] / 100 + 1), 2
            )

            rooms.append({
                "RoomId": room_id,
                "Floor": room_id.split("-")[0],
                "Category": category,
                "Type": room_type,
                "Guests": room_guests,
                "PriceOffSeason": base_price,
                "PricePeakSeason": high_season_price,
            })

    return rooms

def generate_hotel_filename(hotel_key, hotel_name):
    """
    Generate a filename for a hotel based on its key and name.

    Args:
        hotel_key (str): Hotel key
        hotel_name (str): Hotel name

    Returns:
        str: Generated filename
    """
    camel_case_name = ''.join(
        word.capitalize() for word in re.findall(r'\w+', hotel_name)
    )
    return f"hotel_{hotel_key}_{camel_case_name}"

def generate_hotels(config):
    """
    Generate synthetic hotel data.

    Args:
        config (dict): Configuration dictionary

    Returns:
        list: List of generated hotels
    """
    num_hotels = config["num_of_hotels"]
    if num_hotels > 200:
        print(
            "The number of hotels exceeds the limit of 200. "
            "Only 200 hotels will be generated."
        )
        num_hotels = 200
    hotels = []

    config_base_path = os.path.dirname(os.path.dirname(__file__))
    name_location_gen = hotel_name_location_generator.HotelNameLocationGenerator(
        base_path=os.path.join(config_base_path, "../config"),
        config_filename="hotel_naming_location.yaml"
    )

    for _ in range(num_hotels):
        hotel_key = name_location_gen.generate_hotel_key()
        name = name_location_gen.generate_hotel_name()
        address = name_location_gen.generate_address()
        rooms = generate_rooms(config)
        synthetic_parametrization = generate_parametrization(config)

        hotel = {
            "hotelkey": hotel_key,
            "Name": name,
            "Address": address,
            "SyntheticParams": synthetic_parametrization,
            "Rooms": rooms,
        }
        hotels.append(hotel)

    return hotels
