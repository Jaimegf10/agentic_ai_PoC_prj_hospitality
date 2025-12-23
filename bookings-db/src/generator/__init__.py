"""Generator package for creating synthetic hotel and booking data."""

from .hotel_generator import generate_hotels
from .booking_generator import generate_hotel_bookings, all_date_slots, adjust_slots_forecast
from .hotel_query_generator import HotelQueryGenerator
from .hotel_name_location_generator import HotelNameLocationGenerator
from .parametric_utils import *

# Utility functions
from .parametric_utils import (
    get_rooms_floors,
    get_room_type_weights,
    get_room_guests,
    get_room_type_name,
    get_room_category_premium_weight,
    get_room_category,
    get_standard_low_season_prices,
    get_premium_increase,
    get_high_season_increase,
    get_category_price,
    get_hotel_mealplan_weight,
    get_meal_plan,
    get_work_travel,
    get_free_cancellation,
    get_promotion,
    get_non_refundable,
    get_cancellation_fee,
    get_cancellation_status,
    get_number_of_guests,
    get_extra_bed,
    get_meal_plan_prices,
    get_total_price
)

__all__ = [
    # Hotel generation
    'generate_hotels',
    'HotelNameLocationGenerator',

    # Booking generation
    'generate_hotel_bookings',
    'all_date_slots',
    'adjust_slots_forecast',

    # Query generation
    'HotelQueryGenerator',

    # Utility functions
    'get_rooms_floors',
    'get_room_type_weights',
    'get_room_guests',
    'get_room_type_name',
    'get_room_category_premium_weight',
    'get_room_category',
    'get_standard_low_season_prices',
    'get_premium_increase',
    'get_high_season_increase',
    'get_category_price',
    'get_hotel_mealplan_weight',
    'get_meal_plan',
    'get_work_travel',
    'get_free_cancellation',
    'get_promotion',
    'get_non_refundable',
    'get_cancellation_fee',
    'get_cancellation_status',
    'get_number_of_guests',
    'get_extra_bed',
    'get_meal_plan_prices',
    'get_total_price'
]
