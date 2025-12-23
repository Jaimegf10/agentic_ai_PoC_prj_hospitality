"""Module for generating synthetic hotel booking data with realistic patterns and parameters."""

from datetime import timedelta
import random
import calendar
import pandas as pd
from faker import Faker
from . import parametric_utils as ParUt
from . import hotel_name_location_generator

# Name and entity generation
fake = Faker()
Faker.seed(42)
random.seed(42)

def calculate_slot_duration(slot_params: dict, current_date):
    """
    Calculate the duration of a slot based on the current slot count, weekend count, 
    and the current date.

    Parameters:
    - slot_params (dict): Dictionary containing slot parameters:
        - week_count (int): The number of weeks that have been processed so far
        - number_count (int): The total number of slots that have been processed so far
        - weekend_count (int): The number of weekend slots that have been processed so far
        - min_duration (int): The minimum possible duration for a slot
        - max_duration (int): The maximum possible duration for a slot
    - current_date (datetime): The current date for which the slot duration is being 
      calculated

    Returns:
    - int: The calculated duration for the slot

    Logic:
    - For the first 6 slots or if less than 1 week has been processed, the duration is chosen 
      randomly between min_duration and max_duration.
    - If the ratio of weekend slots to total weeks processed is greater than 1, the duration 
      is chosen randomly between min_duration and max_duration.
    - If the current date is before Friday, the duration is chosen randomly between 1 and 4 
      days.
    - If the current date is Friday, the duration is set to 3 days.
    """
    if slot_params['number_count'] < 6 or slot_params['week_count'] < 1:
        slot_duration = random.randint(slot_params['min_duration'], slot_params['max_duration'])
    else:
        if (slot_params['weekend_count']/slot_params['week_count']) > 1:
            slot_duration = random.randint(slot_params['min_duration'], slot_params['max_duration'])
        else:
            if current_date.weekday() < 4:  # Before Friday
                slot_duration = random.randint(1, 4)
            else:  # Friday
                slot_duration = 3  # Duration of 3 days
    return slot_duration

def all_date_slots(start_date: pd.Timestamp,
                end_date: pd.Timestamp,
                min_slot: int = 1,
                max_slot: int = 13):
    """
    Generate a list of date slots between start_date and end_date.

    Parameters:
    - start_date (pd.Timestamp): The start date for slot generation
    - end_date (pd.Timestamp): The end date for slot generation
    - min_slot (int): Minimum slot duration in days
    - max_slot (int): Maximum slot duration in days

    Returns:
    - List[Tuple[pd.Timestamp, pd.Timestamp]]: List of (start, end) date tuples for each slot
    """
    current_date = start_date
    slots = []

    slot_params = {
        'week_count': 0,
        'number_count': 0,
        'weekend_count': 0,
        'min_duration': min_slot,
        'max_duration': max_slot
    }

    while current_date <= end_date:
        max_possible_duration = min(max_slot, (end_date - current_date).days + 1)
        slot_params['max_duration'] = max_possible_duration

        slot_duration = calculate_slot_duration(slot_params, current_date)

        slot_end = current_date + pd.Timedelta(days=slot_duration - 1)

        # Calculate slot end date
        slot_end = min(slot_end, end_date)

        if current_date.weekday() == 4 and slot_duration == 1:
            slot_params['weekend_count'] += 1

        slots.append((current_date, slot_end))

        # Move to the next day after the current slot ends
        current_date = slot_end + pd.Timedelta(days=1)
        slot_params['number_count'] += 1

        # Calculate week count
        days_since_start = (current_date - start_date).days + 1
        slot_params['week_count'] += (days_since_start // 7) - slot_params['week_count']

    # Verify if there are remaining days and add single-day slots to cover them
    remaining_days = (end_date - current_date).days + 1
    for _ in range(remaining_days):
        slots.append((current_date, current_date))
        current_date += pd.Timedelta(days=1)

    return slots


def adjust_slots_occupancy(all_slots, peak_season_months, occupancy_peak, occupancy_offseason):
    """
    Adjust slot occupancy based on peak and off-season parameters.

    Parameters:
    - all_slots (List[Tuple[pd.Timestamp, pd.Timestamp]]): List of date slots
    - peak_season_months (Tuple[str]): Months considered peak season
    - occupancy_peak (int): Target occupancy percentage for peak season
    - occupancy_offseason (int): Target occupancy percentage for off-season

    Returns:
    - List[Tuple[pd.Timestamp, pd.Timestamp]]: Adjusted list of date slots
    """
    slots_by_month = get_slots_by_month(all_slots)
    adjusted_slots = []

    for month, slots in slots_by_month.items():
        # Calculate month parameters
        month_params = {
            'year': int(month.split('-')[0]),
            'month_num': int(month.split('-')[1]),
            'is_peak': calendar.month_name[int(month.split('-')[1])] in peak_season_months,
            'total_days': calendar.monthrange(int(month.split('-')[0]),
                                              int(month.split('-')[1]))[1],
            'current_days': sum((end - start).days + 1 for start, end in slots)
        }
        month_params['target_occupancy'] = (
            occupancy_peak if month_params['is_peak'] else occupancy_offseason
        )
        month_params['target_days'] = int(
            (month_params['target_occupancy'] / 100) * month_params['total_days']
        )

        if month_params['current_days'] > month_params['target_days']:
            # Calculate excess days and slot types
            excess_days = month_params['current_days'] - month_params['target_days']
            slot_types = {
                'non_weekend': [
                    slot for slot in slots
                    if not all(
                        (slot[0] + timedelta(days=i)).weekday() in [4, 5, 6]
                        for i in range((slot[1] - slot[0]).days + 1)
                    )
                ],
                'weekend': [
                    slot for slot in slots
                    if all(
                        (slot[0] + timedelta(days=i)).weekday() in [4, 5, 6]
                        for i in range((slot[1] - slot[0]).days + 1)
                    )
                ]
            }

            # Remove slots to match target occupancy
            while excess_days > 0 and slot_types['non_weekend']:
                slot_to_remove = random.choice(slot_types['non_weekend'])
                slot_types['non_weekend'].remove(slot_to_remove)
                slots.remove(slot_to_remove)
                excess_days -= (slot_to_remove[1] - slot_to_remove[0]).days + 1

            if not month_params['is_peak'] and slot_types['weekend']:
                slot_to_remove = random.choice(slot_types['weekend'])
                slot_types['weekend'].remove(slot_to_remove)
                slots.remove(slot_to_remove)

        adjusted_slots.extend(slots)

    return adjusted_slots


def adjust_slots_forecast(all_slots, current_month, reduce_booking_list):
    """
    Adjust the slots to reduce the occupancy progressively starting from the current month.

    Parameters:
    - all_slots (List[Tuple[pd.Timestamp, pd.Timestamp]]): List of slots.
    - current_month (str): The current month in "YYYY-MM" format.
    - reduce_booking_list (List[int]): List of reduction percentages for each month.

    Returns:
    - List[Tuple[pd.Timestamp, pd.Timestamp]]: Adjusted list of slots.
    """
    slots_by_month = get_slots_by_month(all_slots)
    adjusted_slots, current_month_num, current_year = get_slots_till_current_month(current_month,
                                                                                   slots_by_month)
    adjust_future_slots(slots_by_month, adjusted_slots,
                        current_month_num,
                        current_year,
                        reduce_booking_list)
    return adjusted_slots


def get_slots_by_month(all_slots):
    """
    Group slots by month.

    Parameters:
    - all_slots (List[Tuple[pd.Timestamp, pd.Timestamp]]): List of date slots
    """
    slots_by_month = {}
    for start, end in all_slots:
        month = start.strftime("%Y-%m")
        if month not in slots_by_month:
            slots_by_month[month] = []
        slots_by_month[month].append((start, end))
    return slots_by_month


def get_slots_till_current_month(current_month, slots_by_month):
    """
    Get slots from months before and including the current month.

    Parameters:
    - current_month (str): The current month in "YYYY-MM" format
    - slots_by_month (Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]): 
        Dictionary of slots grouped by month

    Returns:
    - List[Tuple[pd.Timestamp, pd.Timestamp]]: List of slots from months before
        and including the current month
    - int: Current month number
    - int: Current year
    """
    adjusted_slots = []
    current_year, current_month_num = map(int, current_month.split('-'))
    # Add slots from months before and including the current month
    for month, slots in slots_by_month.items():
        year, month_num = map(int, month.split('-'))
        if year < current_year or (year == current_year and month_num <= current_month_num):
            adjusted_slots.extend(slots)
    return adjusted_slots, current_month_num, current_year

def adjust_future_slots(slots_by_month,
                        adjusted_slots,
                        current_month_num,
                        current_year,
                        reduce_booking_list):
    """
    Adjust future slots based on the reduction percentage for each month.

    Parameters:
    - slots_by_month (Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]): 
        Dictionary of slots grouped by month
    - adjusted_slots (List[Tuple[pd.Timestamp, pd.Timestamp]]): List of adjusted slots
    - current_month_num (int): Current month number
    - current_year (int): Current year
    - reduce_booking_list (List[int]): List of reduction percentages for each month
    """
    for i, reduction in enumerate(reduce_booking_list):
        # Calculate month to adjust
        year_to_adjust = current_year + (current_month_num + i - 1) // 12
        month_to_adjust = (current_month_num + i) % 12 + 1
        month_params = {
            'month': month_to_adjust,
            'year': year_to_adjust,
            'str': f"{year_to_adjust:04d}-{month_to_adjust:02d}"
        }

        if month_params['str'] in slots_by_month:
            slots = slots_by_month[month_params['str']]
            total_days = sum((end - start).days + 1 for start, end in slots)
            slot_stats = {
                'total_days': total_days,
                'target_days': int(total_days * (1 - (reduction / 100))),
                'current_days': 0
            }

            if slot_stats['target_days'] <= 0:
                continue

            for slot in slots:
                if slot_stats['current_days'] >= slot_stats['target_days']:
                    break
                slot_days = (slot[1] - slot[0]).days + 1
                if slot_stats['current_days'] + slot_days > slot_stats['target_days']:
                    slot_days = slot_stats['target_days'] - slot_stats['current_days']
                    slot = (slot[0], slot[0] + pd.Timedelta(days=slot_days - 1))
                adjusted_slots.append(slot)
                slot_stats['current_days'] += slot_days

def _generate_guest_info():
    """Generate guest information."""
    name_location_gen = hotel_name_location_generator.HotelNameLocationGenerator()
    guest_location = name_location_gen.generate_guest_location()
    return {
        "FirstName": fake.first_name(),
        "LastName": fake.last_name(),
        "Email": fake.email(),
        "Country": guest_location[0],
        "City": guest_location[1],
        "ZipCode": fake.zipcode(),
        "Address": fake.street_address(),
        "Phone": fake.phone_number()
    }

def _generate_booking_params(room, check_in_date):
    """Generate booking parameters."""
    params = {
        'six_months_before': check_in_date - timedelta(days=6 * 30),
        'room_type_guests': room["Guests"],
        'non_refundable': ParUt.get_non_refundable(),
        'extra_bed': ParUt.get_extra_bed(room["Guests"]),
        'number_of_guests': ParUt.get_number_of_guests(room["Guests"])
    }
    if params['extra_bed'] == "Yes":
        params['number_of_guests'] += 1
    return params

def _create_booking_dict(booking_data: dict):
    """
    Create the complete booking dictionary.

    Parameters:
    - booking_data (dict): Dictionary containing booking information:
        - guest (dict): Guest information
        - params (dict): Booking parameters
        - room (dict): Room information
        - check_in_date (pd.Timestamp): Check-in date
        - check_out_date (pd.Timestamp): Check-out date
        - synthetic_params (dict): Synthetic parameters

    Returns:
    - dict: Complete booking dictionary
    """
    return {
        "ReservationID": str(random.randint(1, 999999)).zfill(6),
        "ReservationDate": fake.date_time_between_dates(
            booking_data['params']['six_months_before'],
            booking_data['check_in_date']
        ).strftime("%Y-%m-%d"),
        "Guest": booking_data['guest'],
        "NumberOfGuests": booking_data['params']['number_of_guests'],
        "ExtraBed": booking_data['params']['extra_bed'],
        "WorkTravel": ParUt.get_work_travel(),
        "CheckInDate": booking_data['check_in_date'].strftime("%Y-%m-%d"),
        "CheckOutDate": booking_data['check_out_date'].strftime("%Y-%m-%d"),
        "RoomAssigned": booking_data['room']["RoomId"],
        "RoomCategory": booking_data['room']["Category"],
        "RoomType": booking_data['room']["Type"],
        "MealPlan": ParUt.get_meal_plan(booking_data['synthetic_params']["MealPlanWeights"]),
        "FreeCancellation": ParUt.get_free_cancellation(),
        "Promotion": ParUt.get_promotion(),
        "NonRefundable": booking_data['params']['non_refundable'],
        "CancellationFee": ParUt.get_cancellation_fee(booking_data['params']['non_refundable']),
        "CancellationStatus": ParUt.get_cancellation_status()
    }

def generate_booking(room, check_in_date, check_out_date, synthetic_params, config):
    """
    Generate a synthetic booking record.

    Parameters:
    - room (dict): Room information
    - check_in_date (pd.Timestamp): Check-in date
    - check_out_date (pd.Timestamp): Check-out date
    - synthetic_params (dict): Synthetic parameters
    - config (dict): Configuration

    Returns:
    - dict: Synthetic booking record
    """
    # Generate guest information and booking parameters
    guest = _generate_guest_info()
    params = _generate_booking_params(room, check_in_date)

    # Create booking dictionary
    booking_data = {
        'guest': guest,
        'params': params,
        'room': room,
        'check_in_date': check_in_date,
        'check_out_date': check_out_date,
        'synthetic_params': synthetic_params
    }
    booking = _create_booking_dict(booking_data)

    # Calculate and add total price
    booking["TotalPrice"] = ParUt.get_total_price(
        booking,  # Use the booking dictionary directly
        room,
        config["peak_season_months"],
        synthetic_params
    )

    return booking


def generate_hotel_bookings(hotel, config):
    """
    Generate synthetic hotel bookings for a given hotel.

    Parameters:
    - hotel (dict): Hotel information
    - config (dict): Configuration

    Returns:
    - dict: Synthetic hotel bookings list
    """
    # Initialize hotel bookings list
    hotel_bookings_list = {
        "HotelKey": hotel["hotelkey"],
        "HotelName": hotel["Name"],
        "Bookings": []
    }

    # Extract configuration parameters
    booking_config = {
        'synthetic_params': hotel["SyntheticParams"],
        'start_year': config["hotel_occupancy"]["booking_year"]["start"],
        'end_year': config["hotel_occupancy"]["booking_year"]["end"],
        'current_month': config["hotel_occupancy"]["current_month"],
        'forecast_reduction': config["hotel_occupancy"]["forecast_reduction_percentage"],
        'peak_season_months': tuple(config["peak_season_months"])
    }

    # Set date range
    date_range = {
        'start': pd.Timestamp(year=booking_config['start_year'], month=1, day=1),
        'end': pd.Timestamp(year=booking_config['end_year'], month=12, day=31)
    }

    # Process each room
    for room in hotel["Rooms"]:
        # Generate and adjust slots
        all_slots = all_date_slots(date_range['start'], date_range['end'])

        adjusted_slots = adjust_slots_occupancy(
            all_slots,
            booking_config['peak_season_months'],
            booking_config['synthetic_params']["OccupancyPeakSeasonWeight"],
            booking_config['synthetic_params']["OccupancyOffSeasonWeight"]
        )

        forecast_adjust_slots = adjust_slots_forecast(
            adjusted_slots,
            booking_config['current_month'],
            booking_config['forecast_reduction']
        )

        # Generate bookings for each slot
        for slot in forecast_adjust_slots:
            check_in_date, check_out_date = slot
            booking = generate_booking(
                room,
                check_in_date,
                check_out_date,
                hotel["SyntheticParams"],
                config
            )
            hotel_bookings_list["Bookings"].append(booking)

    return hotel_bookings_list
