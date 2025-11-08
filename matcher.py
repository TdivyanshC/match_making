#!/usr/bin/env python3
"""
Property Matcher - Terminal-based property matching system
Accepts unstructured requirement messages and matches against inventory
"""

import json
import os
import re
import sys
from typing import Dict, List, Tuple, Any, Optional
import argparse

# Try to import sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Property schema
PROPERTY_SCHEMA = {
    "property_id": None,
    "title": None,
    "raw": None,
    "type": None,  # "rent" or "sale"
    "bhk": None,
    "bathrooms": None,
    "balconies": None,
    "area_sft": None,
    "floor_number": None,
    "total_floors": None,
    "furnishing": None,  # "full", "semi", "none"
    "parking": None,
    "facing": None,
    "amenities": [],
    "location": {"locality": None, "city": None, "landmark": None},
    "rent_rupees": None,
    "total_price_rupees": None,
    "maintenance_monthly": None,
    "negotiable": None,
    "available_from": None,
    "notes": None,
    "contact": {"agent_name": None, "phone": None, "source": None}
}

# Location synonyms
LOCATION_SYNONYMS = {
    "mg road": ["mg rd", "mgroad", "mahatma gandhi road"],
    "ulsoor": ["ulsoor lake", "ulsoor park"],
    "rest house road": ["rest house", "resthouse"],
    "lavelle": ["lavelle road"],
    "noida sec 76": ["noida sector 76", "sector 76"],
    "swiss town": ["swiss town plots"]
}

# Scoring weights
SCORING_WEIGHTS = {
    "budget": 0.4,
    "area": 0.2,
    "location": 0.25,
    "amenities": 0.08,
    "furnishing": 0.04,
    "floor": 0.03
}

class PropertyMatcher:
    def __init__(self, debug=False):
        self.debug = debug
        self.inventory = self.load_inventory()
        self.embeddings_model = None
        self.inventory_embeddings = None
        self.load_embeddings()

    def load_inventory(self) -> List[Dict]:
        """Load inventory from database.json"""
        if os.path.exists('database.json'):
            try:
                with open('database.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        # Create sample inventory if not exists
        sample_inventory = self.create_sample_inventory()
        self.save_inventory(sample_inventory)
        return sample_inventory

    def create_sample_inventory(self) -> List[Dict]:
        """Create sample inventory data"""
        return [
            {
                "property_id": "P001",
                "title": "3BHK Apartment in Ulsoor",
                "raw": "Beautiful 3BHK apartment in Ulsoor, 2400 sqft, semi-furnished, 2.2 lakhs rent",
                "type": "rent",
                "bhk": 3,
                "bathrooms": 2,
                "balconies": 1,
                "area_sft": 2400,
                "floor_number": 5,
                "total_floors": 12,
                "furnishing": "semi",
                "parking": 1,
                "facing": "east",
                "amenities": ["gym", "pool", "garden"],
                "location": {"locality": "Ulsoor", "city": "Bangalore", "landmark": "Ulsoor Lake"},
                "rent_rupees": 220000,
                "maintenance_monthly": 15000,
                "negotiable": True,
                "available_from": "2024-01-15",
                "contact": {"agent_name": "Rajesh Kumar", "phone": "9876543210", "source": "WhatsApp"}
            },
            {
                "property_id": "P002",
                "title": "2BHK in Noida Sector 76",
                "raw": "2BHK apartment near metro, 1200 sqft, furnished, 65,000 rent",
                "type": "rent",
                "bhk": 2,
                "bathrooms": 2,
                "area_sft": 1200,
                "floor_number": 3,
                "total_floors": 15,
                "furnishing": "full",
                "parking": 1,
                "amenities": ["lift", "security"],
                "location": {"locality": "Noida Sector 76", "city": "Noida", "landmark": "Metro Station"},
                "rent_rupees": 65000,
                "maintenance_monthly": 5000,
                "negotiable": False,
                "contact": {"agent_name": "Priya Singh", "phone": "8765432109", "source": "Broker"}
            },
            {
                "property_id": "P003",
                "title": "4BHK Penthouse with Pool",
                "raw": "Luxury 4BHK penthouse, 4000 sqft, pool and gym, 3.5 lakhs rent",
                "type": "rent",
                "bhk": 4,
                "bathrooms": 4,
                "area_sft": 4000,
                "floor_number": 20,
                "total_floors": 20,
                "furnishing": "full",
                "parking": 2,
                "amenities": ["pool", "gym", "terrace", "clubhouse"],
                "location": {"locality": "MG Road", "city": "Bangalore", "landmark": "Near Brigade Road"},
                "rent_rupees": 350000,
                "maintenance_monthly": 25000,
                "negotiable": True,
                "available_from": "2024-02-01",
                "contact": {"agent_name": "Amit Patel", "phone": "7654321098", "source": "Website"}
            },
            {
                "property_id": "P004",
                "title": "3BHK Ground Floor with Garden",
                "raw": "3BHK ground floor apartment with garden, 1800 sqft, semi-furnished, 1.2 lakhs rent",
                "type": "rent",
                "bhk": 3,
                "bathrooms": 2,
                "area_sft": 1800,
                "floor_number": 0,
                "total_floors": 4,
                "furnishing": "semi",
                "parking": 1,
                "amenities": ["garden", "parking"],
                "location": {"locality": "Rest House Road", "city": "Bangalore", "landmark": "Near Lavelle Road"},
                "rent_rupees": 120000,
                "maintenance_monthly": 8000,
                "negotiable": True,
                "contact": {"agent_name": "Sneha Gupta", "phone": "6543210987", "source": "Direct"}
            },
            {
                "property_id": "P005",
                "title": "3BHK for Company Lease",
                "raw": "3BHK apartment for company lease, 2000 sqft, furnished, 1.8 lakhs rent",
                "type": "rent",
                "bhk": 3,
                "bathrooms": 2,
                "area_sft": 2000,
                "floor_number": 8,
                "total_floors": 15,
                "furnishing": "full",
                "parking": 1,
                "amenities": ["lift", "security", "gym"],
                "location": {"locality": "Lavelle Road", "city": "Bangalore", "landmark": "Near MG Road"},
                "rent_rupees": 180000,
                "maintenance_monthly": 12000,
                "negotiable": False,
                "available_from": "2024-10-10",
                "contact": {"agent_name": "Vikram Rao", "phone": "5432109876", "source": "Company"}
            },
            {
                "property_id": "P006",
                "title": "3BHK Budget Option",
                "raw": "3BHK apartment, 1200 sqft, semi-furnished, 55,000 rent",
                "type": "rent",
                "bhk": 3,
                "bathrooms": 2,
                "area_sft": 1200,
                "floor_number": 2,
                "total_floors": 8,
                "furnishing": "semi",
                "parking": 1,
                "amenities": ["parking"],
                "location": {"locality": "Swiss Town", "city": "Bangalore", "landmark": "Near Ring Road"},
                "rent_rupees": 55000,
                "maintenance_monthly": 3000,
                "negotiable": True,
                "contact": {"agent_name": "Kiran Desai", "phone": "4321098765", "source": "Local"}
            }
        ]

    def save_inventory(self, inventory: List[Dict]):
        """Save inventory to database.json"""
        with open('database.json', 'w', encoding='utf-8') as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False)

    def load_embeddings(self):
        """Load or create embeddings for semantic similarity"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            if self.debug:
                print("Sentence transformers not available, using token-based similarity")
            return

        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Check if embeddings file exists
            embeddings_file = 'inventory_embeddings.npy'
            if os.path.exists(embeddings_file):
                self.inventory_embeddings = np.load(embeddings_file)
                if self.debug:
                    print(f"Loaded embeddings from {embeddings_file}")
            else:
                # Create embeddings
                texts = [prop.get('raw', '') for prop in self.inventory]
                self.inventory_embeddings = self.embeddings_model.encode(texts)
                np.save(embeddings_file, self.inventory_embeddings)
                if self.debug:
                    print(f"Created and saved embeddings to {embeddings_file}")

        except Exception as e:
            if self.debug:
                print(f"Error loading embeddings: {e}")
            self.embeddings_model = None

    def normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        # Convert to lowercase
        text = text.lower()

        # Remove emojis and special characters
        text = re.sub(r'[^\w\s+.,-]', '', text)

        # Expand common abbreviations
        expansions = {
            'bhk': 'bedroom',
            'sft': 'sqft',
            'sq ft': 'sqft',
            'lacs': 'lakhs',
            'lac': 'lakhs',
            'l': 'lakhs',
            'cr': 'crore',
            'k': '000',
            'approx': 'approximately'
        }

        for abbr, expansion in expansions.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)

        return text

    def extract_numbers(self, text: str) -> List[int]:
        """Extract all numbers from text"""
        return [int(match) for match in re.findall(r'\d+', text)]

def normalize_text(text: str) -> str:
    """Normalize text for processing"""
    # Convert to lowercase
    text = text.lower()

    # Remove emojis and special characters
    text = re.sub(r'[^\w\s+.,-]', '', text)

    # Expand common abbreviations
    expansions = {
        'bhk': 'bedroom',
        'sft': 'sqft',
        'sq ft': 'sqft',
        'lacs': 'lakhs',
        'lac': 'lakhs',
        'l': 'lakhs',
        'cr': 'crore',
        'k': '000',
        'approx': 'approximately'
    }

    for abbr, expansion in expansions.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)

    return text

def parse_requirement(raw_text: str) -> Dict:
    """Parse unstructured requirement text"""
    text = normalize_text(raw_text)

    requirement = {
        "raw": raw_text,
        "bhk": None,
        "min_area": None,
        "max_area": None,
        "budget_max": None,
        "includes_maintenance": False,
        "preferred_locations": [],
        "hard_constraints": {
            "budget": False,
            "location": False,
            "bhk": False,
            "area": False
        },
        "amenities": {
            "must": [],
            "prefer": [],
            "avoid": []
        },
        "floor_preference": None,  # "higher", "ground", "any"
        "furnishing_preference": None  # "full", "semi", "none"
    }

    # Extract BHK
    bhk_match = re.search(r'(\d+)\s*bedroom', text)
    if bhk_match:
        requirement["bhk"] = int(bhk_match.group(1))
    else:
        # Also try direct BHK pattern
        bhk_match = re.search(r'(\d+)\s*bhk', text)
        if bhk_match:
            requirement["bhk"] = int(bhk_match.group(1))
        else:
            # Try "bed" pattern
            bed_match = re.search(r'(\d+)\s*bed', text)
            if bed_match:
                requirement["bhk"] = int(bed_match.group(1))

    # Extract area constraints
    area_matches = re.findall(r'(\d+)\s*sqft', text)
    if area_matches:
        areas = [int(area) for area in area_matches]
        requirement["min_area"] = min(areas)
        if '+' in text or 'above' in text:
            requirement["min_area"] = min(areas)
        else:
            requirement["max_area"] = max(areas)
    else:
        # Also try 'sft' abbreviation
        area_matches = re.findall(r'(\d+)\s*sft', text)
        if area_matches:
            areas = [int(area) for area in area_matches]
            requirement["min_area"] = min(areas)
            if '+' in text or 'above' in text:
                requirement["min_area"] = min(areas)
            else:
                requirement["max_area"] = max(areas)

    # Extract budget
    budget_patterns = [
        r'max\s*(\d+(?:\.\d+)?)\s*lakhs?',
        r'budget\s*(\d+(?:\.\d+)?)\s*lakhs?',
        r'under\s*(\d+(?:\.\d+)?)\s*lakhs?',
        r'(\d+(?:\.\d+)?)\s*l(?:akhs?)?\b',
        r'(\d+(?:\.\d+)?)\s*l\b',  # Handle "1.5L"
        r'(\d+(?:\.\d+)?)\s*lacs?\b'  # Handle "2lacs"
    ]

    for pattern in budget_patterns:
        match = re.search(pattern, text)
        if match:
            budget = float(match.group(1))
            requirement["budget_max"] = int(budget * 100000)
            break

    # Check if maintenance is included
    if 'including maintenance' in text or 'inc maint' in text or 'with maintenance' in text:
        requirement["includes_maintenance"] = True

    # Extract locations
    for location, synonyms in LOCATION_SYNONYMS.items():
        if location in text or any(syn in text for syn in synonyms):
            requirement["preferred_locations"].append(location)

    # Additional location extraction
    location_keywords = ['ulsoor', 'mg road', 'rest house road', 'lavelle', 'noida sec', 'swiss town']
    for loc in location_keywords:
        if loc in text and loc not in requirement["preferred_locations"]:
            requirement["preferred_locations"].append(loc)

    # Detect hard constraints
    hard_signals = ['must', 'strict', 'only', 'no more than', 'max', 'exactly']
    for signal in hard_signals:
        if signal in text:
            if 'budget' in text or 'lakh' in text:
                requirement["hard_constraints"]["budget"] = True
            if any(loc in text for loc in requirement["preferred_locations"]):
                requirement["hard_constraints"]["location"] = True
            if 'bhk' in text or 'bedroom' in text:
                requirement["hard_constraints"]["bhk"] = True

    # Extract amenities
    amenity_keywords = {
        'must': ['mandatory', 'must have', 'required', 'need'],
        'prefer': ['prefer', 'would like', 'nice to have'],
        'avoid': ['no ', 'not ', 'avoid', 'without']
    }

    amenities_list = ['pool', 'gym', 'garden', 'terrace', 'lift', 'parking', 'clubhouse']

    for amenity in amenities_list:
        if amenity in text:
            if any(signal in text for signal in amenity_keywords['must']):
                requirement["amenities"]["must"].append(amenity)
            elif any(signal in text for signal in amenity_keywords['avoid']):
                requirement["amenities"]["avoid"].append(amenity)
            else:
                requirement["amenities"]["prefer"].append(amenity)

    # Floor preferences
    if 'higher floor' in text or 'high floor' in text:
        requirement["floor_preference"] = "higher"
    elif 'ground' in text and 'garden' in text:
        requirement["floor_preference"] = "ground"

    # Furnishing preferences
    if 'furnished' in text:
        if 'semi' in text:
            requirement["furnishing_preference"] = "semi"
        elif 'fully' in text or 'full' in text:
            requirement["furnishing_preference"] = "full"
    elif 'unfurnished' in text:
        requirement["furnishing_preference"] = "none"

    return requirement

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.embeddings_model and self.inventory_embeddings is not None:
            # Use embeddings
            text_embedding = self.embeddings_model.encode([text1])[0]
            similarities = np.dot(self.inventory_embeddings, text_embedding) / (
                np.linalg.norm(self.inventory_embeddings, axis=1) * np.linalg.norm(text_embedding)
            )
            return float(np.max(similarities))
        else:
            # Token-based fallback
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            return len(intersection) / len(union) if union else 0.0

def score_property(requirement: Dict, property: Dict) -> Tuple[float, Dict]:
    """Score a property against requirements"""
    explanation = {
        "final_score": 0.0,
        "component_scores": {
            "budget": 0.0,
            "area": 0.0,
            "location": 0.0,
            "amenities": 0.0,
            "furnishing": 0.0,
            "floor": 0.0
        },
        "why_included": [],
        "why_excluded": []
    }

    # Hard filters
    excluded = False

    # BHK hard constraint
    if requirement.get("hard_constraints", {}).get("bhk", False):
        if requirement.get("bhk") and property.get("bhk") != requirement["bhk"]:
            explanation["why_excluded"].append(f"BHK mismatch: required {requirement['bhk']}, property has {property.get('bhk')}")
            excluded = True

    # Budget hard constraint
    if requirement.get("hard_constraints", {}).get("budget", False):
        max_budget = requirement.get("budget_max")
        property_price = property.get("rent_rupees") or property.get("total_price_rupees", 0)

        if max_budget and property_price > max_budget * 1.3:  # Allow 30% over budget
            explanation["why_excluded"].append(f"Over budget: required max {max_budget}, property costs {property_price}")
            excluded = True

    # Location hard constraint
    if requirement.get("hard_constraints", {}).get("location", False):
        req_locations = requirement.get("preferred_locations", [])
        prop_locality = property.get("location", {}).get("locality", "").lower()

        location_match = any(
            req_loc.lower() in prop_locality or
            any(syn.lower() in prop_locality for syn in LOCATION_SYNONYMS.get(req_loc, []))
            for req_loc in req_locations
        )

        if not location_match:
            explanation["why_excluded"].append(f"Location mismatch: required {req_locations}, property in {prop_locality}")
            excluded = True

    if excluded:
        return 0.0, explanation

    # Soft scoring components

    # Budget score
    max_budget = requirement.get("budget_max")
    property_price = property.get("rent_rupees") or property.get("total_price_rupees", 0)

    if max_budget and property_price:
        if property_price <= max_budget:
            budget_score = 1.0
            explanation["why_included"].append("Within budget")
        else:
            # Decaying score for over-budget properties
            over_budget_ratio = property_price / max_budget
            budget_score = max(0.0, 1.0 - (over_budget_ratio - 1.0) * 0.5)
            explanation["why_included"].append(f"Slightly over budget ({over_budget_ratio:.1f}x)")
    else:
        budget_score = 0.5  # Neutral if no budget specified

    explanation["component_scores"]["budget"] = budget_score

    # Area score
    min_area = requirement.get("min_area")
    property_area = property.get("area_sft", 0)

    if min_area and property_area:
        if property_area >= min_area:
            area_score = min(1.0, property_area / min_area)
            explanation["why_included"].append(f"Good area: {property_area} sqft >= {min_area} sqft")
        else:
            area_score = property_area / min_area * 0.5  # Penalty for smaller area
            explanation["why_included"].append(f"Smaller area: {property_area} sqft < {min_area} sqft")
    else:
        area_score = 0.5

    explanation["component_scores"]["area"] = area_score

    # Location score
    req_locations = requirement.get("preferred_locations", [])
    prop_locality = property.get("location", {}).get("locality", "").lower()

    location_score = 0.0
    for req_loc in req_locations:
        if req_loc.lower() in prop_locality:
            location_score = 1.0
            explanation["why_included"].append(f"Exact location match: {req_loc}")
            break
        elif any(syn.lower() in prop_locality for syn in LOCATION_SYNONYMS.get(req_loc, [])):
            location_score = 0.9
            explanation["why_included"].append(f"Location synonym match: {req_loc}")
            break
        else:
            # Semantic similarity
            similarity = calculate_semantic_similarity(req_loc, prop_locality)
            location_score = max(location_score, similarity * 0.8)

    if location_score > 0:
        explanation["component_scores"]["location"] = location_score
    else:
        explanation["component_scores"]["location"] = 0.2  # Base score

    # Amenities score
    must_amenities = requirement.get("amenities", {}).get("must", [])
    prefer_amenities = requirement.get("amenities", {}).get("prefer", [])
    avoid_amenities = requirement.get("amenities", {}).get("avoid", [])
    property_amenities = [amenity.lower() for amenity in property.get("amenities", [])]

    amenities_score = 0.0

    # Check must-have amenities
    if must_amenities:
        missing_must = [amenity for amenity in must_amenities if amenity.lower() not in property_amenities]
        if missing_must:
            explanation["why_excluded"].append(f"Missing required amenities: {missing_must}")
            return 0.0, explanation
        else:
            amenities_score += 0.5
            explanation["why_included"].append(f"Has all required amenities: {must_amenities}")

    # Check preferred amenities
    if prefer_amenities:
        matched_prefer = [amenity for amenity in prefer_amenities if amenity.lower() in property_amenities]
        if matched_prefer:
            amenities_score += len(matched_prefer) / len(prefer_amenities) * 0.3
            explanation["why_included"].append(f"Has preferred amenities: {matched_prefer}")

    # Check avoided amenities
    if avoid_amenities:
        has_avoided = [amenity for amenity in avoid_amenities if amenity.lower() in property_amenities]
        if has_avoided:
            explanation["why_excluded"].append(f"Has avoided amenities: {has_avoided}")
            return 0.0, explanation

    explanation["component_scores"]["amenities"] = amenities_score

    # Furnishing score
    req_furnishing = requirement.get("furnishing_preference")
    prop_furnishing = property.get("furnishing")

    if req_furnishing and prop_furnishing:
        if req_furnishing == prop_furnishing:
            furnishing_score = 1.0
            explanation["why_included"].append(f"Furnishing match: {req_furnishing}")
        else:
            furnishing_score = 0.5
    else:
        furnishing_score = 0.5

    explanation["component_scores"]["furnishing"] = furnishing_score

    # Floor preference score
    req_floor = requirement.get("floor_preference")
    prop_floor = property.get("floor_number", 0)

    if req_floor == "higher" and prop_floor >= 5:
        floor_score = 1.0
        explanation["why_included"].append("Higher floor as preferred")
    elif req_floor == "ground" and prop_floor == 0:
        floor_score = 1.0
        explanation["why_included"].append("Ground floor with garden")
    else:
        floor_score = 0.5

    explanation["component_scores"]["floor"] = floor_score

    # Calculate final score
    final_score = (
        SCORING_WEIGHTS["budget"] * budget_score +
        SCORING_WEIGHTS["area"] * area_score +
        SCORING_WEIGHTS["location"] * location_score +
        SCORING_WEIGHTS["amenities"] * amenities_score +
        SCORING_WEIGHTS["furnishing"] * furnishing_score +
        SCORING_WEIGHTS["floor"] * floor_score
    )

    explanation["final_score"] = final_score * 100  # Convert to percentage

    return final_score, explanation

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.embeddings_model and self.inventory_embeddings is not None:
            # Use embeddings
            text_embedding = self.embeddings_model.encode([text1])[0]
            similarities = np.dot(self.inventory_embeddings, text_embedding) / (
                np.linalg.norm(self.inventory_embeddings, axis=1) * np.linalg.norm(text_embedding)
            )
            return float(np.max(similarities))
        else:
            # Token-based fallback
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            return len(intersection) / len(union) if union else 0.0

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts"""
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        # Use embeddings if available
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception:
            pass

    # Token-based fallback
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0.0

    def score_property(self, requirement: Dict, property: Dict) -> Tuple[float, Dict]:
        explanation = {
            "final_score": 0.0,
            "component_scores": {
                "budget": 0.0,
                "area": 0.0,
                "location": 0.0,
                "amenities": 0.0,
                "furnishing": 0.0,
                "floor": 0.0
            },
            "why_included": [],
            "why_excluded": []
        }

        # Hard filters
        excluded = False

        # BHK hard constraint
        if requirement.get("hard_constraints", {}).get("bhk", False):
            if requirement.get("bhk") and property.get("bhk") != requirement["bhk"]:
                explanation["why_excluded"].append(f"BHK mismatch: required {requirement['bhk']}, property has {property.get('bhk')}")
                excluded = True

        # Budget hard constraint
        if requirement.get("hard_constraints", {}).get("budget", False):
            max_budget = requirement.get("budget_max")
            property_price = property.get("rent_rupees") or property.get("total_price_rupees", 0)

            if max_budget and property_price > max_budget * 1.3:  # Allow 30% over budget
                explanation["why_excluded"].append(f"Over budget: required max {max_budget}, property costs {property_price}")
                excluded = True

        # Location hard constraint
        if requirement.get("hard_constraints", {}).get("location", False):
            req_locations = requirement.get("preferred_locations", [])
            prop_locality = property.get("location", {}).get("locality", "").lower()

            location_match = any(
                req_loc.lower() in prop_locality or
                any(syn.lower() in prop_locality for syn in LOCATION_SYNONYMS.get(req_loc, []))
                for req_loc in req_locations
            )

            if not location_match:
                explanation["why_excluded"].append(f"Location mismatch: required {req_locations}, property in {prop_locality}")
                excluded = True

        if excluded:
            return 0.0, explanation

        # Soft scoring components

        # Budget score
        max_budget = requirement.get("budget_max")
        property_price = property.get("rent_rupees") or property.get("total_price_rupees", 0)

        if max_budget and property_price:
            if property_price <= max_budget:
                budget_score = 1.0
                explanation["why_included"].append("Within budget")
            else:
                # Decaying score for over-budget properties
                over_budget_ratio = property_price / max_budget
                budget_score = max(0.0, 1.0 - (over_budget_ratio - 1.0) * 0.5)
                explanation["why_included"].append(f"Slightly over budget ({over_budget_ratio:.1f}x)")
        else:
            budget_score = 0.5  # Neutral if no budget specified

        explanation["component_scores"]["budget"] = budget_score

        # Area score
        min_area = requirement.get("min_area")
        property_area = property.get("area_sft", 0)

        if min_area and property_area:
            if property_area >= min_area:
                area_score = min(1.0, property_area / min_area)
                explanation["why_included"].append(f"Good area: {property_area} sqft >= {min_area} sqft")
            else:
                area_score = property_area / min_area * 0.5  # Penalty for smaller area
                explanation["why_included"].append(f"Smaller area: {property_area} sqft < {min_area} sqft")
        else:
            area_score = 0.5

        explanation["component_scores"]["area"] = area_score

        # Location score
        req_locations = requirement.get("preferred_locations", [])
        prop_locality = property.get("location", {}).get("locality", "").lower()

        location_score = 0.0
        for req_loc in req_locations:
            if req_loc.lower() in prop_locality:
                location_score = 1.0
                explanation["why_included"].append(f"Exact location match: {req_loc}")
                break
            elif any(syn.lower() in prop_locality for syn in LOCATION_SYNONYMS.get(req_loc, [])):
                location_score = 0.9
                explanation["why_included"].append(f"Location synonym match: {req_loc}")
                break
            else:
                # Semantic similarity
                similarity = self.calculate_semantic_similarity(req_loc, prop_locality)
                location_score = max(location_score, similarity * 0.8)

        if location_score > 0:
            explanation["component_scores"]["location"] = location_score
        else:
            explanation["component_scores"]["location"] = 0.2  # Base score

        # Amenities score
        must_amenities = requirement.get("amenities", {}).get("must", [])
        prefer_amenities = requirement.get("amenities", {}).get("prefer", [])
        avoid_amenities = requirement.get("amenities", {}).get("avoid", [])
        property_amenities = [amenity.lower() for amenity in property.get("amenities", [])]

        amenities_score = 0.0

        # Check must-have amenities
        if must_amenities:
            missing_must = [amenity for amenity in must_amenities if amenity.lower() not in property_amenities]
            if missing_must:
                explanation["why_excluded"].append(f"Missing required amenities: {missing_must}")
                return 0.0, explanation
            else:
                amenities_score += 0.5
                explanation["why_included"].append(f"Has all required amenities: {must_amenities}")

        # Check preferred amenities
        if prefer_amenities:
            matched_prefer = [amenity for amenity in prefer_amenities if amenity.lower() in property_amenities]
            if matched_prefer:
                amenities_score += len(matched_prefer) / len(prefer_amenities) * 0.3
                explanation["why_included"].append(f"Has preferred amenities: {matched_prefer}")

        # Check avoided amenities
        if avoid_amenities:
            has_avoided = [amenity for amenity in avoid_amenities if amenity.lower() in property_amenities]
            if has_avoided:
                explanation["why_excluded"].append(f"Has avoided amenities: {has_avoided}")
                return 0.0, explanation

        explanation["component_scores"]["amenities"] = amenities_score

        # Furnishing score
        req_furnishing = requirement.get("furnishing_preference")
        prop_furnishing = property.get("furnishing")

        if req_furnishing and prop_furnishing:
            if req_furnishing == prop_furnishing:
                furnishing_score = 1.0
                explanation["why_included"].append(f"Furnishing match: {req_furnishing}")
            else:
                furnishing_score = 0.5
        else:
            furnishing_score = 0.5

        explanation["component_scores"]["furnishing"] = furnishing_score

        # Floor preference score
        req_floor = requirement.get("floor_preference")
        prop_floor = property.get("floor_number", 0)

        if req_floor == "higher" and prop_floor >= 5:
            floor_score = 1.0
            explanation["why_included"].append("Higher floor as preferred")
        elif req_floor == "ground" and prop_floor == 0:
            floor_score = 1.0
            explanation["why_included"].append("Ground floor with garden")
        else:
            floor_score = 0.5

        explanation["component_scores"]["floor"] = floor_score

        # Calculate final score
        final_score = (
            SCORING_WEIGHTS["budget"] * budget_score +
            SCORING_WEIGHTS["area"] * area_score +
            SCORING_WEIGHTS["location"] * location_score +
            SCORING_WEIGHTS["amenities"] * amenities_score +
            SCORING_WEIGHTS["furnishing"] * furnishing_score +
            SCORING_WEIGHTS["floor"] * floor_score
        )

        explanation["final_score"] = final_score * 100  # Convert to percentage

        return final_score, explanation

    def find_matches(self, requirement_text: str) -> List[Tuple[Dict, float, Dict]]:
        """Find and score matching properties"""
        requirement = parse_requirement(requirement_text)

        matches = []
        for property in self.inventory:
            score, explanation = score_property(requirement, property)
            if score > 0:  # Include all properties with any score
                matches.append((property, score, explanation))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    def display_results(self, matches: List[Tuple[Dict, float, Dict]]):
        """Display matching results"""
        if not matches:
            print("No matching properties found.")
            return

        # Check for high-score matches (>= 90%)
        high_score_matches = [match for match in matches if match[1] >= 0.9]

        if high_score_matches:
            display_matches = high_score_matches
            print(f"\nFound {len(display_matches)} excellent matches (>=90%):")
        else:
            display_matches = matches[:3]  # Top 3
            print(f"\nTop {len(display_matches)} matches:")

        print("=" * 80)

        for i, (property, score, explanation) in enumerate(display_matches, 1):
            print(f"\n{i}. {property['title']}")
            print(f"   Score: {score*100:.1f}%")

            # Property details
            details = []
            if property.get('bhk'):
                details.append(f"{property['bhk']}BHK")
            if property.get('area_sft'):
                details.append(f"{property['area_sft']} sqft")
            if property.get('type') == 'rent' and property.get('rent_rupees'):
                details.append(f"Rent: ₹{property['rent_rupees']:,}")
            elif property.get('total_price_rupees'):
                details.append(f"Price: ₹{property['total_price_rupees']:,}")

            location = property.get('location', {})
            if location.get('locality'):
                details.append(f"Location: {location['locality']}")

            print(f"   Details: {', '.join(details)}")

            # Why recommended
            print("   Why recommended:")
            for reason in explanation.get('why_included', []):
                print(f"   • {reason}")

            if explanation.get('why_excluded'):
                print("   Concerns:")
                for concern in explanation['why_excluded']:
                    print(f"   • {concern}")

            print(f"   Contact: {property.get('contact', {}).get('agent_name', 'N/A')} - {property.get('contact', {}).get('phone', 'N/A')}")

        # Save results to JSON
        results_data = {
            "requirement": requirement_text,
            "matches": [
                {
                    "property": prop,
                    "score": score,
                    "explanation": exp
                }
                for prop, score, exp in display_matches
            ]
        }

        with open('last_query_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to last_query_results.json")

    def contact_agent_prompt(self, matches: List[Tuple[Dict, float, Dict]]):
        """Prompt user to contact agent"""
        if not matches:
            return

        while True:
            choice = input("\nWould you like to contact an agent? (y/n): ").strip().lower()
            if choice == 'y':
                print("\nSelect a property to contact (1-{}):".format(len(matches)))
                for i, (prop, _, _) in enumerate(matches, 1):
                    print(f"{i}. {prop['title']} - {prop.get('contact', {}).get('agent_name')}")

                try:
                    selection = int(input("Enter property number: ")) - 1
                    if 0 <= selection < len(matches):
                        prop = matches[selection][0]
                        contact = prop.get('contact', {})
                        agent_name = contact.get('agent_name', 'Agent')
                        phone = contact.get('phone', 'N/A')

                        message = f"""Hi {agent_name},

I'm interested in your property: {prop['title']}
Location: {prop.get('location', {}).get('locality', 'N/A')}
Price: ₹{prop.get('rent_rupees') or prop.get('total_price_rupees', 'N/A'):,}

Please share more details.

Thank you!"""

                        print(f"\nSuggested message to {agent_name} ({phone}):")
                        print("-" * 50)
                        print(message)
                        print("-" * 50)
                        print(f"You can copy this message and send it to {phone}")
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Please enter a valid number.")

                break
            elif choice == 'n':
                break
            else:
                print("Please enter 'y' or 'n'.")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Property Matcher')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    matcher = PropertyMatcher(debug=args.debug)

    print("\n" + "="*60)
    print("🏠 PROPERTY MATCHER - WhatsApp Style Property Search")
    print("="*60)
    print("Paste your property requirements below (WhatsApp style messages)")
    print("Examples:")
    print("• '3BHK, 2400sqft+, near Ulsoor or MG Road. Max 2 lakhs including maintenance. Semi-furnished.'")
    print("• 'Looking for 2BHK under 75L in Noida Sec 76, close to metro. Furnished ok.'")
    print("• '4BHK penthouse, 4000sft, pool + gym mandatory, budget 3.5 lacs (rental).'")
    print("• '3 bhk, budget ~1.5L. No lift, prefer ground with garden.'")
    print("• '3bhk for company lease, prefer Rest House Road / Lavelle. Budget 2lacs'")
    print("• 'Cheap 3bed, 1200sft, 60L, negotiable, near Swiss Town'")
    print("\nType 'menu' for options, 'exit' to quit")
    print("-"*60)

    while True:
        try:
            print("\n📝 Paste your property requirement:")
            requirement_text = input().strip()

            if not requirement_text:
                continue

            if requirement_text.lower() == 'exit':
                print("👋 Goodbye!")
                break

            if requirement_text.lower() == 'menu':
                print("\n" + "="*40)
                print("OPTIONS MENU")
                print("="*40)
                print("1. Run tests")
                print("2. Show help")
                print("3. Exit")
                print("-"*40)

                choice = input("Enter choice (1-3): ").strip()
                if choice == '1':
                    print("Running tests...")
                    run_tests()
                    print("Tests completed.")
                elif choice == '2':
                    print("Help: Just paste your property requirements above. The system will parse and match automatically.")
                elif choice == '3':
                    print("👋 Goodbye!")
                    break
                else:
                    print("Invalid choice.")
                continue

            # Process the requirement
            print("\n🔄 Processing your requirement...")
            matches = matcher.find_matches(requirement_text)
            matcher.display_results(matches)
            matcher.contact_agent_prompt(matches)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

def run_tests():
    """Run unit tests"""
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_matcher.py', '-v'],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Could not run tests: {e}")

if __name__ == "__main__":
    main()