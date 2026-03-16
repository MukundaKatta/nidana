"""Twenty medical specialties for the Nidana benchmark."""

from enum import Enum


class MedicalSpecialty(str, Enum):
    """Twenty medical specialties spanning the breadth of clinical medicine."""

    CARDIOLOGY = "cardiology"
    PULMONOLOGY = "pulmonology"
    GASTROENTEROLOGY = "gastroenterology"
    NEPHROLOGY = "nephrology"
    NEUROLOGY = "neurology"
    ENDOCRINOLOGY = "endocrinology"
    RHEUMATOLOGY = "rheumatology"
    HEMATOLOGY_ONCOLOGY = "hematology_oncology"
    INFECTIOUS_DISEASE = "infectious_disease"
    DERMATOLOGY = "dermatology"
    PSYCHIATRY = "psychiatry"
    OBSTETRICS_GYNECOLOGY = "obstetrics_gynecology"
    PEDIATRICS = "pediatrics"
    EMERGENCY_MEDICINE = "emergency_medicine"
    ORTHOPEDICS = "orthopedics"
    UROLOGY = "urology"
    OPHTHALMOLOGY = "ophthalmology"
    OTOLARYNGOLOGY = "otolaryngology"
    ALLERGY_IMMUNOLOGY = "allergy_immunology"
    GENERAL_SURGERY = "general_surgery"

    @property
    def display_name(self) -> str:
        """Human-readable specialty name."""
        return _DISPLAY_NAMES[self]

    @property
    def description(self) -> str:
        """Brief clinical description of the specialty scope."""
        return _DESCRIPTIONS[self]


_DISPLAY_NAMES: dict[MedicalSpecialty, str] = {
    MedicalSpecialty.CARDIOLOGY: "Cardiology",
    MedicalSpecialty.PULMONOLOGY: "Pulmonology",
    MedicalSpecialty.GASTROENTEROLOGY: "Gastroenterology",
    MedicalSpecialty.NEPHROLOGY: "Nephrology",
    MedicalSpecialty.NEUROLOGY: "Neurology",
    MedicalSpecialty.ENDOCRINOLOGY: "Endocrinology",
    MedicalSpecialty.RHEUMATOLOGY: "Rheumatology",
    MedicalSpecialty.HEMATOLOGY_ONCOLOGY: "Hematology-Oncology",
    MedicalSpecialty.INFECTIOUS_DISEASE: "Infectious Disease",
    MedicalSpecialty.DERMATOLOGY: "Dermatology",
    MedicalSpecialty.PSYCHIATRY: "Psychiatry",
    MedicalSpecialty.OBSTETRICS_GYNECOLOGY: "Obstetrics & Gynecology",
    MedicalSpecialty.PEDIATRICS: "Pediatrics",
    MedicalSpecialty.EMERGENCY_MEDICINE: "Emergency Medicine",
    MedicalSpecialty.ORTHOPEDICS: "Orthopedics",
    MedicalSpecialty.UROLOGY: "Urology",
    MedicalSpecialty.OPHTHALMOLOGY: "Ophthalmology",
    MedicalSpecialty.OTOLARYNGOLOGY: "Otolaryngology (ENT)",
    MedicalSpecialty.ALLERGY_IMMUNOLOGY: "Allergy & Immunology",
    MedicalSpecialty.GENERAL_SURGERY: "General Surgery",
}

_DESCRIPTIONS: dict[MedicalSpecialty, str] = {
    MedicalSpecialty.CARDIOLOGY: (
        "Disorders of the heart and cardiovascular system including coronary artery disease, "
        "heart failure, arrhythmias, valvular disease, and congenital heart defects."
    ),
    MedicalSpecialty.PULMONOLOGY: (
        "Diseases of the respiratory system including asthma, COPD, interstitial lung disease, "
        "pulmonary embolism, and lung infections."
    ),
    MedicalSpecialty.GASTROENTEROLOGY: (
        "Disorders of the gastrointestinal tract, liver, pancreas, and biliary system including "
        "inflammatory bowel disease, hepatitis, and GI malignancies."
    ),
    MedicalSpecialty.NEPHROLOGY: (
        "Kidney diseases including acute and chronic kidney injury, glomerulonephritis, "
        "electrolyte disorders, and renal replacement therapy."
    ),
    MedicalSpecialty.NEUROLOGY: (
        "Disorders of the central and peripheral nervous system including stroke, epilepsy, "
        "multiple sclerosis, movement disorders, and neurodegenerative diseases."
    ),
    MedicalSpecialty.ENDOCRINOLOGY: (
        "Hormonal and metabolic disorders including diabetes mellitus, thyroid disease, "
        "adrenal disorders, pituitary disease, and calcium metabolism."
    ),
    MedicalSpecialty.RHEUMATOLOGY: (
        "Autoimmune and inflammatory disorders of joints, connective tissue, and musculoskeletal "
        "system including rheumatoid arthritis, lupus, and vasculitis."
    ),
    MedicalSpecialty.HEMATOLOGY_ONCOLOGY: (
        "Blood disorders and malignancies including anemia, coagulopathies, leukemias, "
        "lymphomas, and solid-organ cancers."
    ),
    MedicalSpecialty.INFECTIOUS_DISEASE: (
        "Bacterial, viral, fungal, and parasitic infections including HIV, tuberculosis, "
        "endocarditis, and emerging infections."
    ),
    MedicalSpecialty.DERMATOLOGY: (
        "Skin, hair, and nail disorders including psoriasis, eczema, skin cancers, "
        "autoimmune blistering diseases, and cutaneous infections."
    ),
    MedicalSpecialty.PSYCHIATRY: (
        "Mental health disorders including mood disorders, psychotic disorders, anxiety, "
        "substance use disorders, and personality disorders."
    ),
    MedicalSpecialty.OBSTETRICS_GYNECOLOGY: (
        "Reproductive health, pregnancy, and female pelvic disorders including preeclampsia, "
        "ectopic pregnancy, and gynecologic malignancies."
    ),
    MedicalSpecialty.PEDIATRICS: (
        "Medical care of infants, children, and adolescents including congenital disorders, "
        "developmental milestones, and childhood infections."
    ),
    MedicalSpecialty.EMERGENCY_MEDICINE: (
        "Acute and life-threatening conditions requiring urgent evaluation and stabilization "
        "including trauma, sepsis, and acute coronary syndromes."
    ),
    MedicalSpecialty.ORTHOPEDICS: (
        "Musculoskeletal injuries and diseases including fractures, joint disorders, "
        "sports injuries, and spinal pathology."
    ),
    MedicalSpecialty.UROLOGY: (
        "Disorders of the urinary tract and male reproductive system including kidney stones, "
        "prostate disease, and urologic malignancies."
    ),
    MedicalSpecialty.OPHTHALMOLOGY: (
        "Eye diseases including glaucoma, macular degeneration, diabetic retinopathy, "
        "cataracts, and acute vision loss."
    ),
    MedicalSpecialty.OTOLARYNGOLOGY: (
        "Ear, nose, and throat disorders including hearing loss, sinusitis, head and neck "
        "cancers, and airway management."
    ),
    MedicalSpecialty.ALLERGY_IMMUNOLOGY: (
        "Allergic diseases and immune system disorders including anaphylaxis, asthma, "
        "primary immunodeficiencies, and drug hypersensitivity."
    ),
    MedicalSpecialty.GENERAL_SURGERY: (
        "Operative management of abdominal, endocrine, breast, and soft-tissue diseases "
        "including appendicitis, hernias, and surgical oncology."
    ),
}
