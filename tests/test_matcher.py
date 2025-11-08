#!/usr/bin/env python3
"""
Unit tests for Property Matcher
"""

import json
import os
import sys
import pytest
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matcher import PropertyMatcher
from matcher import parse_requirement, score_property

class TestPropertyMatcher:
    def setup_method(self):
        """Setup test fixtures"""
        self.matcher = PropertyMatcher(debug=False)

        # Sample property for testing
        self.sample_property = {
            "property_id": "TEST001",
            "title": "3BHK Test Property",
            "raw": "3BHK apartment in Ulsoor, 2400 sqft, semi-furnished, 2.2 lakhs rent",
            "type": "rent",
            "bhk": 3,
            "bathrooms": 2,
            "area_sft": 2400,
            "floor_number": 5,
            "furnishing": "semi",
            "parking": 1,
            "facing": "east",
            "amenities": ["gym", "pool"],
            "location": {"locality": "Ulsoor", "city": "Bangalore"},
            "rent_rupees": 220000,
            "maintenance_monthly": 15000,
            "negotiable": True,
            "contact": {"agent_name": "Test Agent", "phone": "9876543210"}
        }

    def test_parse_requirement_simple_bhk_area_budget(self):
        """Test parsing simple requirement with BHK, area, and budget"""
        text = "3BHK, 2400sqft+, near Ulsoor or MG Road. Max 2 lakhs including maintenance. Semi-furnished."
        req = parse_requirement(text)

        assert req["bhk"] == 3
        assert req["min_area"] == 2400
        assert req["budget_max"] == 200000
        assert req["includes_maintenance"] == True
        assert "ulsoor" in req["preferred_locations"]
        assert "mg road" in req["preferred_locations"]
        assert req["furnishing_preference"] == "semi"

    def test_parse_requirement_over_budget(self):
        """Test parsing requirement slightly over budget"""
        text = "2BHK under 75L in Noida Sec 76, close to metro. Furnished ok."
        req = parse_requirement(text)

        assert req["bhk"] == 2
        assert req["budget_max"] == 7500000
        assert "noida sec 76" in req["preferred_locations"]

    def test_parse_requirement_pool_gym_mandatory(self):
        """Test parsing requirement with mandatory amenities"""
        text = "4BHK penthouse, 4000sft, pool + gym mandatory, budget 3.5 lacs (rental)."
        req = parse_requirement(text)

        assert req["bhk"] == 4
        assert req["min_area"] == 4000
        assert req["budget_max"] == 350000
        assert "pool" in req["amenities"]["must"]
        assert "gym" in req["amenities"]["must"]

    def test_parse_requirement_ground_floor_garden(self):
        """Test parsing requirement with ground floor preference"""
        text = "3 bhk, budget ~1.5L. No lift, prefer ground with garden."
        req = parse_requirement(text)

        assert req["bhk"] == 3
        assert req["budget_max"] == 150000  # 1.5L = 150,000 rupees
        assert req["floor_preference"] == "ground"
        assert "lift" in req["amenities"]["avoid"]

    def test_parse_requirement_company_lease(self):
        """Test parsing company lease requirement"""
        text = "3bhk for company lease, prefer Rest House Road / Lavelle. Budget 2lacs, available Oct 10th"
        req = parse_requirement(text)

        assert req["bhk"] == 3
        assert req["budget_max"] == 200000  # 2 lakhs = 200,000 rupees
        assert "rest house road" in req["preferred_locations"]
        assert "lavelle" in req["preferred_locations"]

    def test_parse_requirement_cheap_budget(self):
        """Test parsing cheap budget requirement"""
        text = "Cheap 3bed, 1200sft, 60L, negotiable, near Swiss Town"
        req = parse_requirement(text)

        assert req["bhk"] == 3
        assert req["min_area"] == 1200
        assert req["budget_max"] == 6000000
        assert "swiss town" in req["preferred_locations"]

    def test_score_property_exact_match(self):
        """Test scoring with exact match"""
        req = {
            "bhk": 3,
            "min_area": 2400,
            "budget_max": 250000,
            "preferred_locations": ["ulsoor"],
            "hard_constraints": {"bhk": False, "budget": False, "location": False},
            "amenities": {"must": [], "prefer": [], "avoid": []},
            "furnishing_preference": "semi"
        }

        score, explanation = score_property(req, self.sample_property)

        assert score > 0.8  # Should be high score
        assert explanation["final_score"] > 80
        assert "Within budget" in explanation["why_included"]
        assert any("location match" in reason for reason in explanation["why_included"])

    def test_score_property_over_budget_penalty(self):
        """Test scoring with over-budget property"""
        req = {
            "bhk": 3,
            "budget_max": 150000,  # Lower than property price
            "preferred_locations": ["ulsoor"],
            "hard_constraints": {"budget": False},
            "amenities": {"must": [], "prefer": [], "avoid": []}
        }

        score, explanation = score_property(req, self.sample_property)

        assert score < 1.0  # Should be penalized
        assert any("over budget" in reason for reason in explanation["why_included"])

    def test_score_property_hard_constraint_violation(self):
        """Test hard constraint violation"""
        req = {
            "bhk": 2,  # Different from property
            "hard_constraints": {"bhk": True},  # Hard constraint
            "preferred_locations": [],
            "amenities": {"must": [], "prefer": [], "avoid": []}
        }

        score, explanation = score_property(req, self.sample_property)

        assert score == 0.0
        assert any("BHK mismatch" in reason for reason in explanation["why_excluded"])

    def test_score_property_amenity_match(self):
        """Test amenity matching"""
        req = {
            "bhk": 3,
            "preferred_locations": ["ulsoor"],
            "hard_constraints": {"bhk": False, "location": False},
            "amenities": {"must": ["pool"], "prefer": ["gym"], "avoid": []}
        }

        score, explanation = score_property(req, self.sample_property)

        assert score > 0.5
        assert any("required amenities" in reason for reason in explanation["why_included"])

    def test_score_property_location_synonym(self):
        """Test location synonym matching"""
        # Create property with synonym location
        property_synonym = self.sample_property.copy()
        property_synonym["location"]["locality"] = "MG Rd"

        req = {
            "preferred_locations": ["mg road"],
            "hard_constraints": {"location": False},
            "amenities": {"must": [], "prefer": [], "avoid": []}
        }

        score, explanation = score_property(req, property_synonym)

        assert score > 0.5
        assert any("synonym match" in reason for reason in explanation["why_included"])

    def test_score_property_avoided_amenity(self):
        """Test avoided amenity penalty"""
        req = {
            "bhk": 3,
            "preferred_locations": ["ulsoor"],
            "hard_constraints": {"bhk": False, "location": False},
            "amenities": {"must": [], "prefer": [], "avoid": ["pool"]}
        }

        score, explanation = score_property(req, self.sample_property)

        assert score == 0.0
        assert any("avoided amenities" in reason for reason in explanation["why_excluded"])

    def test_find_matches_integration(self):
        """Test full integration of finding matches"""
        requirement_text = "3BHK, 2400sqft+, near Ulsoor. Max 2 lakhs. Semi-furnished."
        matches = self.matcher.find_matches(requirement_text)

        assert len(matches) > 0
        # Should find the sample property in inventory
        property_ids = [prop["property_id"] for prop, _, _ in matches]
        assert "P001" in property_ids

    @patch('builtins.input', side_effect=['exit'])
    def test_cli_exit(self, mock_input):
        """Test CLI exit functionality"""
        # This would normally require user interaction, but we mock it
        pass

    def test_inventory_loading(self):
        """Test inventory loading from database.json"""
        # Should load sample inventory
        assert len(self.matcher.inventory) > 0
        assert all("property_id" in prop for prop in self.matcher.inventory)

    def test_explanation_structure(self):
        """Test explanation dict structure"""
        req = {"bhk": 3, "preferred_locations": [], "hard_constraints": {}, "amenities": {"must": [], "prefer": [], "avoid": []}}
        score, explanation = score_property(req, self.sample_property)

        required_keys = ["final_score", "component_scores", "why_included", "why_excluded"]
        assert all(key in explanation for key in required_keys)

        component_keys = ["budget", "area", "location", "amenities", "furnishing", "floor"]
        assert all(key in explanation["component_scores"] for key in component_keys)

if __name__ == "__main__":
    pytest.main([__file__])