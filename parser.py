# Property Data Ingestion Parser
# Extracts structured data from unstructured property text

import re
import json
import os
from typing import Dict, Any, Optional, List

PROPERTY_SCHEMA = {
    "property_id": None,
    "title": None,
    "description": None,
    "property_type": None,
    "bhk": None,
    "bathrooms": None,
    "balconies": None,
    "super_area_sqft": None,
    "carpet_area_sqft": None,
    "floor_number": None,
    "total_floors": None,
    "furnishing": None,
    "parking": None,
    "parking_type": None,
    "age_of_property_years": None,
    "ownership": None,
    "location": {
        "city": None,
        "locality": None,
        "landmark": None,
        "latitude": None,
        "longitude": None
    },
    "amenities": [],
    "facing": None,
    "price": {
        "expected_price": None,
        "negotiable": None,
        "maintenance_monthly": None
    },
    "contact": {
        "agent_name": None,
        "phone": None,
        "source": None
    }
}

def extract_numerical_value(text: str, pattern: str) -> Optional[int]:
    """Extract numerical value using regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_float_value(text: str, pattern: str) -> Optional[float]:
    """Extract float value using regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def convert_price_to_inr(price_text: str) -> Optional[int]:
    """Convert price text (Lakh/Crore) to INR."""
    price_text = price_text.lower().strip()

    # Extract numerical value
    match = re.search(r'(\d+(?:\.\d+)?)', price_text)
    if not match:
        return None

    value = float(match.group(1))

    if 'crore' in price_text:
        return int(value * 10000000)
    elif 'lakh' in price_text or 'lac' in price_text:
        return int(value * 100000)
    else:
        # Assume it's already in rupees
        return int(value)

def extract_furnishing(text: str) -> Optional[str]:
    """Extract furnishing status."""
    text_lower = text.lower()
    if 'fully furnished' in text_lower or 'fully-furnished' in text_lower:
        return 'Fully Furnished'
    elif 'semi furnished' in text_lower or 'semi-furnished' in text_lower:
        return 'Semi Furnished'
    elif 'unfurnished' in text_lower:
        return 'Unfurnished'
    return None

def extract_amenities(text: str) -> list:
    """Extract amenities from text."""
    amenities_keywords = [
        'gym', 'swimming pool', 'clubhouse', 'lift', 'elevator', 'parking',
        'security', 'power backup', 'intercom', 'garden', 'playground',
        'temple', 'shopping center', 'hospital', 'school'
    ]
    found_amenities = []
    text_lower = text.lower()
    for amenity in amenities_keywords:
        if amenity in text_lower:
            found_amenities.append(amenity.title())
    return found_amenities

def extract_facing(text: str) -> Optional[str]:
    """Extract property facing."""
    facing_keywords = ['east', 'west', 'north', 'south', 'north-east', 'north-west', 'south-east', 'south-west']
    text_lower = text.lower()
    for face in facing_keywords:
        if face.replace('-', ' ') in text_lower or face in text_lower:
            return face.title().replace('-', ' ')
    return None

def parse_property_specs(text: str) -> Dict[str, Any]:
    """Parse property specifications."""
    specs = {}

    # BHK
    specs['bhk'] = extract_numerical_value(text, r'(\d+)\s*bhk')

    # Bathrooms
    specs['bathrooms'] = extract_numerical_value(text, r'(\d+)\s*bath')

    # Balconies
    specs['balconies'] = extract_numerical_value(text, r'(\d+)\s*balcon')

    # Super Area
    specs['super_area_sqft'] = extract_numerical_value(text, r'(\d+)\s*(?:sqft|sq\.ft|square feet)')

    # Carpet Area
    specs['carpet_area_sqft'] = extract_numerical_value(text, r'carpet.*?(\d+)\s*(?:sqft|sq\.ft)')

    # Floor Number
    specs['floor_number'] = extract_numerical_value(text, r'(\d+)(?:st|nd|rd|th)\s*floor')

    # Total Floors
    specs['total_floors'] = extract_numerical_value(text, r'total\s*(\d+)\s*floors?')

    # Furnishing
    specs['furnishing'] = extract_furnishing(text)

    # Parking
    specs['parking'] = 'parking' in text.lower()

    # Age of property
    specs['age_of_property_years'] = extract_numerical_value(text, r'(\d+)\s*years?\s*old')

    # Facing
    specs['facing'] = extract_facing(text)

    return specs

def parse_location(text: str) -> Dict[str, Any]:
    """Parse location information."""
    location = {}

    # City (common Indian cities)
    cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'pune', 'hyderabad', 'kolkata', 'ahmedabad']
    text_lower = text.lower()
    for city in cities:
        if city in text_lower:
            location['city'] = city.title()
            break

    # Locality (extract after city or common patterns)
    locality_patterns = [
        r'in\s+([A-Za-z\s]+?)(?:\s*,|\s*delhi|\s*mumbai|\s*bangalore|$)',
        r'at\s+([A-Za-z\s]+?)(?:\s*,|\s*near|$)',
        r'([A-Za-z\s]+?)\s*locality'
    ]

    for pattern in locality_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            location['locality'] = match.group(1).strip()
            break

    # Landmark
    landmark_match = re.search(r'near\s+([A-Za-z\s]+?)(?:\s*,|\s*\.|$)', text, re.IGNORECASE)
    if landmark_match:
        location['landmark'] = landmark_match.group(1).strip()

    return location

def parse_price(text: str) -> Dict[str, Any]:
    """Parse pricing information."""
    price_info = {}

    # Expected price
    price_patterns = [
        r'(?:rs\.?|₹|price)\s*(\d+(?:\.\d+)?)\s*(?:lakh|lacs?|crore)',
        r'(\d+(?:\.\d+)?)\s*(?:lakh|lacs?|crore)',
        r'(?:rs\.?|₹)\s*(\d+(?:,\d+)*)'
    ]

    for pattern in price_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            price_text = match.group(0)
            price_info['expected_price'] = convert_price_to_inr(price_text)
            break

    # Negotiable
    price_info['negotiable'] = 'negotiable' in text.lower()

    # Maintenance
    maintenance_match = re.search(r'maintenance\s*(?:rs\.?|₹)?\s*(\d+)', text, re.IGNORECASE)
    if maintenance_match:
        price_info['maintenance_monthly'] = int(maintenance_match.group(1))

    return price_info

def parse_contact(text: str) -> Dict[str, Any]:
    """Parse contact information."""
    contact = {}

    # Phone number
    phone_match = re.search(r'(\+91[\s-]?)?(\d{10}|\d{5}[\s-]\d{5})', text)
    if phone_match:
        contact['phone'] = phone_match.group(0).replace(' ', '').replace('-', '')

    # Agent name (common patterns)
    name_patterns = [
        r'contact\s+([A-Za-z\s]+)',
        r'call\s+([A-Za-z\s]+)',
        r'agent\s+([A-Za-z\s]+)',
        r'([A-Za-z\s]+)\s*\(\s*agent\s*\)'
    ]

    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            contact['agent_name'] = match.group(1).strip()
            break

    # Source
    if 'whatsapp' in text.lower():
        contact['source'] = 'WhatsApp'
    elif 'broker' in text.lower():
        contact['source'] = 'Broker'
    elif 'website' in text.lower():
        contact['source'] = 'Website'

    return contact

def parse_property_text(text: str) -> Dict[str, Any]:
    """Main function to parse unstructured property text."""
    # Start with schema template
    property_data = PROPERTY_SCHEMA.copy()
    property_data['location'] = PROPERTY_SCHEMA['location'].copy()
    property_data['price'] = PROPERTY_SCHEMA['price'].copy()
    property_data['contact'] = PROPERTY_SCHEMA['contact'].copy()

    # Set description
    property_data['description'] = text

    # Extract title (first line or first sentence)
    lines = text.split('\n')
    property_data['title'] = lines[0].strip() if lines else text[:100]

    # Parse different sections
    specs = parse_property_specs(text)
    property_data.update(specs)

    location = parse_location(text)
    property_data['location'].update(location)

    price = parse_price(text)
    property_data['price'].update(price)

    contact = parse_contact(text)
    property_data['contact'].update(contact)

    # Amenities
    property_data['amenities'] = extract_amenities(text)

    # Property type (basic detection)
    if 'apartment' in text.lower() or 'flat' in text.lower():
        property_data['property_type'] = 'Apartment'
    elif 'villa' in text.lower():
        property_data['property_type'] = 'Villa'
    elif 'plot' in text.lower():
        property_data['property_type'] = 'Plot'

    # Generate property ID if not found
    if not property_data['property_id']:
        property_data['property_id'] = f"P{hash(text) % 10000:04d}"

    return property_data

def load_database() -> List[Dict[str, Any]]:
    """Load existing database from file."""
    if os.path.exists('database.json'):
        try:
            with open('database.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_to_database(property_data: Dict[str, Any]) -> bool:
    """Save property data to database.json."""
    try:
        database = load_database()
        database.append(property_data)

        with open('database.json', 'w', encoding='utf-8') as f:
            json.dump(database, f, indent=2, ensure_ascii=False, default=str)

        return True
    except Exception as e:
        print(f"Error saving to database: {e}")
        return False

def display_menu():
    """Display the main menu."""
    print("\n" + "="*50)
    print("PROPERTY DATA INGESTION SYSTEM")
    print("="*50)
    print("Paste your property details below (or type 'exit' to quit)")
    print("Supported formats: WhatsApp messages, broker notes, website text, etc.")
    print("-" * 50)

def get_user_input() -> Optional[str]:
    """Get property details from user."""
    print("\nPaste property details here:")
    print("(Press Enter twice when finished)")
    print()

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        except EOFError:
            break

    text = "\n".join(lines).strip()
    return text if text else None

def main():
    """Interactive CLI for property data ingestion."""
    while True:
        display_menu()

        property_text = get_user_input()

        if not property_text:
            print("No input provided. Please try again.")
            continue

        if property_text.lower().strip() == 'exit':
            print("Goodbye!")
            break

        # Parse the property text
        print("\n🔄 Parsing property data...")
        parsed_data = parse_property_text(property_text)

        # Display parsed JSON
        print("\nParsed Property Data:")
        print("=" * 50)
        print(json.dumps(parsed_data, indent=2, ensure_ascii=False, default=str))
        print("=" * 50)

        # Ask to save
        while True:
            save_choice = input("\nDo you want to save this property to database? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                if save_to_database(parsed_data):
                    print("Property saved to database.json successfully!")
                else:
                    print("Failed to save property.")
                break
            elif save_choice in ['n', 'no']:
                print("Property not saved.")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

        # Ask to continue
        while True:
            continue_choice = input("\nProcess another property? (y/n): ").strip().lower()
            if continue_choice in ['y', 'yes']:
                break
            elif continue_choice in ['n', 'no']:
                print("Goodbye!")
                return
            else:
                print("Please enter 'y' for yes or 'n' for no.")

if __name__ == "__main__":
    main()