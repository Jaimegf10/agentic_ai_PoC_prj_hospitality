# HOWTO: Synthetic Hotel Data Generator

## 📋 Description

`gen_synthetic_hotels.py` is a tool for generating synthetic hotel, room, and booking data. This tool is useful for populating test databases, developing hospitality applications, or generating training datasets for AI models.

## 📁 Location

```
bookings-db/src/gen_synthetic_hotels.py
```

## 🔧 Dependencies

### Installing dependencies

```bash
cd bookings-db
pip install -r requirements.txt
```

### Required libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `PyYAML` | >=6.0.0 | Reading YAML configuration files |
| `pandas` | >=2.0.0 | Data manipulation and CSV/Excel export |
| `openpyxl` | 3.2.0b1 | Excel file generation (.xlsx) |
| `Faker` | 36.1.1 | Fake data generation (names, addresses, etc.) |
| `numpy` | 2.2.3 | Numerical operations for room distribution |
| `python-dateutil` | 2.9.0.post0 | Date handling for bookings |
| `pytz` / `tzdata` | 2025.1 | Timezone support |

## 🚀 How to Run

### Basic execution

```bash
cd bookings-db/src
python gen_synthetic_hotels.py
```

### From the project root directory

```bash
cd bookings-db
python -m src.gen_synthetic_hotels
```

## ⚙️ Configuration Files

The tool uses two YAML configuration files located in `bookings-db/config/`:

### 1. `generate_hotels_param.yaml` - Generation Configuration

This file controls all synthetic data generation parameters:

#### General Configuration

```yaml
num_of_hotels: 5                           # Number of hotels to generate
process:
  output_path_hotels: output/hotels/       # Output path for hotel data
  output_path_bookings: output/bookings/   # Output path for bookings
```

#### Peak Season Months

```yaml
peak_season_months:
  - January
  - April
  - May
  - June
  - July
  - August
  - September
  - December
```

Months not listed are considered off-season.

#### Room Configuration

```yaml
rooms_per_hotel:
  number:
    min: 30                    # Minimum rooms per hotel
    max: 80                    # Maximum rooms per hotel
  floors:
    min: 1                     # Minimum floors
    max: 3                     # Maximum floors
  weight_triple_rooms:         # Triple room weight
    min: 10
    max: 30
  weight_double_rooms:         # Double room weight
    default: True              # Default type
    min: 50
    max: 70
  weight_single_rooms:         # Single room weight
    min: 20
    max: 30
  weight_premium_rooms:        # Premium room weight
    min: 10
    max: 40
```

#### Pricing Configuration

```yaml
pricing:
  single_room_standard_low_season:
    min: 40                              # Minimum single room price (€)
    max: 80                              # Maximum single room price (€)
  double_room_standard_low_season:
    min: 50
    max: 100
  triple_room_standard_low_season:
    min: 100
    max: 150
  premium_price_increase_percentage:     # Premium category increase
    min: 50
    max: 150
  peak_season_price_increase_percentage: # Peak season increase
    min: 50
    max: 100
  promotion_price_discount_percentage:   # Promotional discount
    min: 10
    max: 30
```

#### Occupancy and Booking Configuration

```yaml
hotel_occupancy:
  booking_year:
    start: 2025
    end: 2025
  current_month: 2025-04              # Current month for forecast calculations
  forecast_reduction_percentage:       # Progressive forecast reduction
    - 15
    - 30
    - 45
    - 60
    - 75
    - 90
    - 100
  occupancy_weight:
    peak_season:
      min: 75                         # Minimum peak season occupancy (%)
      max: 100                        # Maximum peak season occupancy (%)
    off_season:
      min: 20                         # Minimum off-season occupancy (%)
      max: 40                         # Maximum off-season occupancy (%)
```

#### Meal Plans

```yaml
meal_plans_weight:
  room_only:
    name: "Room Only"
    min: 10
    max: 20
    price_increase_percentage:
      min: 0
      max: 0
  room_and_breakfast:
    default: True
    name: "Room and Breakfast"
    min: 30
    max: 60
    price_increase_percentage:
      min: 10
      max: 20
  all_inclusive:
    name: "All Inclusive"
    min: 0
    max: 10
    price_increase_percentage:
      min: 100
      max: 120
  half_board:
    name: "Half Board"
    min: 10
    max: 30
    price_increase_percentage:
      min: 30
      max: 50
  full_board:
    name: "Full Board"
    min: 10
    max: 30
    price_increase_percentage:
      min: 50
      max: 70
```

### 2. `hotel_queries.yaml` - Query Templates

This file defines query templates for generating hotel question datasets. It contains three categories:

- **hotel_detail_queries**: Queries about hotel details (addresses, meal charges, discounts)
- **room_queries**: Queries about rooms and pricing
- **hotel_bookings_queries**: Queries about reservations

Templates use placeholders like `{hotel_name}`, `{city}`, `{country}` which are replaced with actual data.

## 📤 Output Files

The tool generates the following files:

### Directory `output/hotels/`

| File | Format | Description |
|------|--------|-------------|
| `hotels.json` | JSON | Complete hotel data with nested structure |
| `hotels.xlsx` | Excel | Hotel data in tabular format |
| `hotels.csv` | CSV | Hotel data for processing |
| `all_hotels.csv` | CSV | Consolidated data for all hotels |
| `hotel_details.md` | Markdown | Human-readable hotel and room documentation |
| `hotel_rooms.md` | Markdown | Room table per hotel |
| `hotel_room_queries.csv` | CSV | Generated queries dataset |

### Directory `output/bookings/`

| File | Format | Description |
|------|--------|-------------|
| `all_bookings.xlsx` | Excel | All bookings from all hotels |

Additionally, in `output/hotels/`:
- `hotel_bookings.md` - Markdown with all bookings

## 📊 Generated Data Structure

### Hotel Structure (JSON)

```json
{
  "hotelkey": "1825",
  "Name": "Obsidian Tower",
  "Address": {
    "Country": "France",
    "City": "Cannes",
    "ZipCode": "84311",
    "Address": "43321 Brittany Bypass"
  },
  "SyntheticParams": {
    "OccupancyPeakSeasonWeight": 83,
    "OccupancyOffSeasonWeight": 40,
    "OccupancyBaseDiscountPercentage": 25,
    "ExtraBedChargePercentage": 21,
    "MealPlanWeights": {...},
    "MealPlanPrices": {...},
    "PromotionPriceDiscount": 30
  },
  "Rooms": [
    {
      "RoomId": "01-001",
      "Floor": "01",
      "Category": "Standard",
      "Type": "Double",
      "Guests": 2,
      "PriceOffSeason": 115.5,
      "PricePeakSeason": 179.03
    }
  ]
}
```

### Booking Structure

```json
{
  "RoomAssigned": "01-001",
  "RoomType": "Double",
  "RoomCategory": "Standard",
  "CheckInDate": "2025-01-15",
  "CheckOutDate": "2025-01-18",
  "Guest": {
    "FirstName": "John",
    "LastName": "Smith",
    "Email": "john.smith@email.com",
    "Phone": "+33123456789",
    "Country": "Germany",
    "City": "Berlin",
    "Address": "123 Main Street",
    "ZipCode": "10115"
  },
  "MealPlan": "Room and Breakfast",
  "TotalPrice": 534.09
}
```

## 🏗️ Module Architecture

```
bookings-db/src/
├── gen_synthetic_hotels.py          # Main script
├── generator/                        # Generation modules
│   ├── hotel_generator.py           # Generates hotel data
│   ├── booking_generator.py         # Generates bookings
│   ├── hotel_name_location_generator.py  # Generates names and locations
│   ├── hotel_query_generator.py     # Generates test queries
│   └── parametric_utils.py          # Parameter utilities
└── output/                          # Writer modules
    ├── hotel_output_writer.py       # Writes hotel files
    ├── booking_output_writer.py     # Writes booking files
    └── hotel_query_writer.py        # Writes queries
```

## 💡 Usage Examples

### Generate 10 hotels

Modify `generate_hotels_param.yaml`:

```yaml
num_of_hotels: 10
```

### Change peak season pricing

```yaml
pricing:
  peak_season_price_increase_percentage:
    min: 100    # +100% minimum
    max: 150    # +150% maximum
```

### Adjust occupancy rates

```yaml
hotel_occupancy:
  occupancy_weight:
    peak_season:
      min: 90
      max: 100
    off_season:
      min: 40
      max: 60
```

## ⏱️ Execution Time

The script displays total execution time upon completion:

```
Synthetic Data successfully generated 3.45 sg
```

Time varies depending on:
- Number of configured hotels
- Room range per hotel
- Booking period (booking_year)

## 🔍 Troubleshooting

### Error: FileNotFoundError on config

Make sure to run from the correct directory:

```bash
cd bookings-db/src
python gen_synthetic_hotels.py
```

### Error: ModuleNotFoundError

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Output files are not generated

Verify that output directories exist or that the script has write permissions:

```bash
mkdir -p output/hotels output/bookings
```

## 📝 Additional Notes

- Prices are in **euros (€)**
- Generated hotels are located in **France** (cities like Paris, Nice, Cannes, etc.)
- Guest names and addresses are generated using the **Faker** library
- The tool is **non-deterministic** by default (no fixed random seed)
