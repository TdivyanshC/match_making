# Property Matchmaking System
# A terminal-based application to match user property requirements with an inventory of plots.

import sys
import re

# Parsed inventory data from unstructured input
inventory = [
    {
        "id": "GR001",
        "location": "Godrej Reserve",
        "size_sqft": 1200,
        "facing": "East and North",
        "budget": 1200 * 8700
    },
    {
        "id": "GR002",
        "location": "Godrej Reserve",
        "size_sqft": 1500,
        "facing": "East/North/North East",
        "budget": 1500 * 8700
    },
    {
        "id": "GR003",
        "location": "Godrej Reserve",
        "size_sqft": 1800,
        "facing": "East",
        "budget": 1800 * 8700
    },
    {
        "id": "GR004",
        "location": "Godrej Reserve",
        "size_sqft": 2400,
        "facing": "West East",
        "budget": 2400 * 8800
    },
    {
        "id": "GR005",
        "location": "Godrej Reserve",
        "size_sqft": 2850,
        "facing": "West",
        "budget": 2850 * 8500
    },
    {
        "id": "GR006",
        "location": "Godrej Reserve",
        "size_sqft": 3200,
        "facing": "North",
        "budget": 3200 * 9200
    },
    {
        "id": "GR007",
        "location": "Godrej Reserve",
        "size_sqft": 3200,
        "facing": "North",
        "budget": 3200 * 9700
    },
    {
        "id": "GR008",
        "location": "Godrej Reserve",
        "size_sqft": 1200,
        "facing": "East",
        "budget": 1200 * 9200
    },
    {
        "id": "PPD001",
        "location": "Prestige Park Drive",
        "size_sqft": 1200,
        "facing": "West",
        "budget": 1200 * 10000
    },
    {
        "id": "PPD002",
        "location": "Prestige Park Drive",
        "size_sqft": 1200,
        "facing": "West",
        "budget": 1200 * 12000
    },
    {
        "id": "PT001",
        "location": "Purva Tivoli",
        "size_sqft": 1200,
        "facing": "East",
        "budget": 1200 * 8300
    },
    {
        "id": "ST001",
        "location": "Swiss Town",
        "size_sqft": 4550,
        "facing": "West",
        "budget": 4550 * 9750
    }
]

def display_menu():
    """Display the main menu options."""
    print("\n----------------------------")
    print("1. Search Property")
    print("2. Show All Properties")
    print("3. Exit")
    print("----------------------------")

def get_user_choice():
    """Get and validate user menu choice."""
    try:
        choice = int(input("Enter your choice (1-3): "))
        if choice in [1, 2, 3]:
            return choice
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None

def extract_requirements(message):
    """Extract requirements from unstructured message."""
    req = {
        "location": None,
        "size_sqft": None,
        "budget": None,
        "facing": None
    }

    # Extract location (simple keyword matching)
    locations = ["godrej reserve", "prestige park drive", "purva tivoli", "swiss town", "north bangalore"]
    for loc in locations:
        if loc.lower() in message.lower():
            req["location"] = loc.title()
            break

    # Extract size (numbers followed by sqft)
    size_match = re.search(r'(\d+)\s*sqft', message.lower())
    if size_match:
        req["size_sqft"] = int(size_match.group(1))

    # Extract budget (numbers, possibly with lakhs or per sqft)
    budget_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?)', message.lower())
    if budget_match:
        req["budget"] = int(float(budget_match.group(1)) * 100000)
    else:
        # Check for per sqft pricing
        per_sqft_match = re.search(r'(\d+)\s*per\s*sqft', message.lower())
        if per_sqft_match and req["size_sqft"]:
            req["budget"] = req["size_sqft"] * int(per_sqft_match.group(1))

    # Extract facing
    facing_keywords = ["east", "west", "north", "south"]
    for face in facing_keywords:
        if face in message.lower():
            req["facing"] = face.title()
            break

    return req

def calculate_match_score(property, req_location, req_size, req_budget, req_facing):
    """Calculate match score out of 100 based on requirements."""
    score = 0

    # Location: 40% - partial match
    if req_location and req_location.lower() in property["location"].lower():
        score += 40
    elif req_location:
        # Partial match for similar locations
        if "godrej" in req_location.lower() and "godrej" in property["location"].lower():
            score += 30
        elif "prestige" in req_location.lower() and "prestige" in property["location"].lower():
            score += 30

    # Size: 30% - higher if >= requirement
    if req_size and property["size_sqft"] >= req_size:
        size_score = min(100, (property["size_sqft"] / req_size) * 100)
        score += 0.3 * size_score

    # Budget: 20% - higher if <= requirement
    if req_budget and property["budget"] <= req_budget:
        budget_score = min(100, (req_budget / property["budget"]) * 100)
        score += 0.2 * budget_score

    # Facing: 10% - exact match
    if req_facing and req_facing.lower() in property["facing"].lower():
        score += 10

    return round(score)

def search_properties():
    """Handle property search based on unstructured user message."""
    print("\n--- Search Property ---")
    print("Enter your requirements in natural language (e.g., 'I need a 1200 sqft plot in Godrej Reserve with east facing under 10 lakhs'):")
    message = input().strip()

    req = extract_requirements(message)
    print(f"\nExtracted requirements: Location={req['location']}, Size={req['size_sqft']} sqft, Budget=₹{req['budget']}, Facing={req['facing']}")

    # Calculate scores for all properties
    matches = []
    for prop in inventory:
        score = calculate_match_score(prop, req['location'], req['size_sqft'], req['budget'], req['facing'])
        matches.append((prop, score))

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)

    # Filter top matches
    top_matches = [m for m in matches if m[1] >= 80]  # Lower threshold for plots
    if not top_matches:
        top_matches = matches[:5]  # Show more for plots

    # Display results
    if top_matches:
        print("\n==================================")
        print("✅ Top Matches For Your Requirement")
        for prop, score in top_matches:
            print(f"ID: {prop['id']}")
            print(f"Location: {prop['location']}")
            print(f"Size: {prop['size_sqft']} SqFt | Facing: {prop['facing']}")
            print(f"Total Price: ₹{prop['budget']:,}")
            print(f"Match Score: {score}%")
            print("==================================")
    else:
        print("No matching properties found.")

def show_all_properties():
    """Display all properties in the inventory."""
    print("\n--- All Properties ---")
    print("==================================")
    for prop in inventory:
        print(f"ID: {prop['id']}")
        print(f"Location: {prop['location']}")
        print(f"Size: {prop['size_sqft']} SqFt | Facing: {prop['facing']}")
        print(f"Total Price: ₹{prop['budget']:,}")
        print("==================================")

def main():
    """Main program loop."""
    while True:
        display_menu()
        choice = get_user_choice()
        if choice == 1:
            search_properties()
        elif choice == 2:
            show_all_properties()
        elif choice == 3:
            print("Exiting the application. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()