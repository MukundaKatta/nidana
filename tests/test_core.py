"""Tests for Nidana."""
from src.core import Nidana
def test_init(): assert Nidana().get_stats()["ops"] == 0
def test_op(): c = Nidana(); c.process(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Nidana(); [c.process() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Nidana(); c.process(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Nidana(); r = c.process(); assert r["service"] == "nidana"
