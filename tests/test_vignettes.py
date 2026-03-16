"""Tests for the clinical vignette module."""

from __future__ import annotations

import pytest

from nidana.vignettes.generator import ClinicalVignette, Vitals, VignetteBank
from nidana.vignettes.specialties import MedicalSpecialty


class TestMedicalSpecialty:
    """Tests for the MedicalSpecialty enum."""

    def test_has_twenty_specialties(self) -> None:
        assert len(MedicalSpecialty) == 20

    def test_display_names_exist(self) -> None:
        for sp in MedicalSpecialty:
            assert sp.display_name, f"Missing display_name for {sp}"

    def test_descriptions_exist(self) -> None:
        for sp in MedicalSpecialty:
            assert sp.description, f"Missing description for {sp}"
            assert len(sp.description) > 20, f"Description too short for {sp}"

    def test_enum_values_are_lowercase(self) -> None:
        for sp in MedicalSpecialty:
            assert sp.value == sp.value.lower()


class TestClinicalVignette:
    """Tests for the ClinicalVignette model."""

    def test_create_basic_vignette(self) -> None:
        v = ClinicalVignette(
            specialty=MedicalSpecialty.CARDIOLOGY,
            patient_age=55,
            patient_sex="M",
            chief_complaint="chest pain",
            history_of_present_illness="Acute onset substernal chest pain.",
            correct_diagnosis="STEMI",
        )
        assert v.patient_age == 55
        assert v.specialty == MedicalSpecialty.CARDIOLOGY
        assert v.id  # auto-generated

    def test_to_prompt_contains_key_sections(self) -> None:
        v = ClinicalVignette(
            specialty=MedicalSpecialty.NEUROLOGY,
            patient_age=70,
            patient_sex="F",
            chief_complaint="sudden onset weakness",
            history_of_present_illness="Right-sided weakness for 2 hours.",
            past_medical_history="Atrial fibrillation",
            vitals=Vitals(heart_rate=90, blood_pressure="180/100"),
            labs="Glucose 120 mg/dL",
            correct_diagnosis="Acute ischemic stroke",
            differential=["Hemorrhagic stroke", "Todd paralysis"],
        )
        prompt = v.to_prompt()
        assert "70-year-old female" in prompt
        assert "sudden onset weakness" in prompt
        assert "Atrial fibrillation" in prompt
        assert "HR 90 bpm" in prompt
        assert "Glucose 120" in prompt
        assert "most likely diagnosis" in prompt

    def test_vitals_formatting(self) -> None:
        v = Vitals(heart_rate=100, blood_pressure="120/80", spo2=95)
        from nidana.vignettes.generator import _format_vitals

        text = _format_vitals(v)
        assert "HR 100 bpm" in text
        assert "BP 120/80" in text
        assert "SpO2 95%" in text


class TestVignetteBank:
    """Tests for the built-in vignette bank."""

    @pytest.fixture
    def bank(self) -> VignetteBank:
        return VignetteBank()

    def test_bank_has_at_least_40_vignettes(self, bank: VignetteBank) -> None:
        assert len(bank) >= 40, f"Expected >=40 vignettes, got {len(bank)}"

    def test_bank_covers_at_least_10_specialties(self, bank: VignetteBank) -> None:
        specialties = bank.specialties_with_vignettes()
        assert len(specialties) >= 10, f"Expected >=10 specialties, got {len(specialties)}"

    def test_each_covered_specialty_has_at_least_3_vignettes(self, bank: VignetteBank) -> None:
        for sp in bank.specialties_with_vignettes():
            count = len(bank.by_specialty(sp))
            assert count >= 3, (
                f"Specialty {sp.display_name} has only {count} vignettes (need >=3)"
            )

    def test_all_vignettes_have_required_fields(self, bank: VignetteBank) -> None:
        for v in bank.all:
            assert v.patient_age > 0
            assert v.patient_sex in ("M", "F")
            assert len(v.chief_complaint) > 5
            assert len(v.history_of_present_illness) > 20
            assert len(v.correct_diagnosis) > 3
            assert len(v.differential) >= 2, f"Vignette {v.id} needs >=2 differentials"

    def test_all_vignettes_have_dangerous_misses(self, bank: VignetteBank) -> None:
        for v in bank.all:
            assert len(v.dangerous_misses) >= 1, (
                f"Vignette {v.id} ({v.correct_diagnosis}) needs dangerous_misses"
            )

    def test_all_vignettes_produce_valid_prompts(self, bank: VignetteBank) -> None:
        for v in bank.all:
            prompt = v.to_prompt()
            assert len(prompt) > 100
            assert "most likely diagnosis" in prompt

    def test_by_specialty_filter(self, bank: VignetteBank) -> None:
        cardio = bank.by_specialty(MedicalSpecialty.CARDIOLOGY)
        assert all(v.specialty == MedicalSpecialty.CARDIOLOGY for v in cardio)

    def test_difficulty_values_are_valid(self, bank: VignetteBank) -> None:
        valid = {"easy", "moderate", "hard"}
        for v in bank.all:
            assert v.difficulty in valid, f"Invalid difficulty '{v.difficulty}' in {v.id}"

    def test_vignette_ids_are_unique(self, bank: VignetteBank) -> None:
        ids = [v.id for v in bank.all]
        assert len(ids) == len(set(ids)), "Duplicate vignette IDs found"
