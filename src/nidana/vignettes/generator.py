"""Clinical vignette models and the built-in vignette bank."""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field

from nidana.vignettes.specialties import MedicalSpecialty


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class Vitals(BaseModel):
    """Patient vital signs."""

    heart_rate: Optional[int] = Field(None, description="Beats per minute")
    blood_pressure: Optional[str] = Field(None, description="Systolic/diastolic mmHg")
    respiratory_rate: Optional[int] = Field(None, description="Breaths per minute")
    temperature_c: Optional[float] = Field(None, description="Degrees Celsius")
    spo2: Optional[int] = Field(None, description="Oxygen saturation percentage")
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None


class ClinicalVignette(BaseModel):
    """A single clinical case used for benchmarking LLM diagnostic reasoning."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    specialty: MedicalSpecialty
    difficulty: str = Field(
        default="moderate",
        description="easy | moderate | hard",
    )
    patient_age: int
    patient_sex: str = Field(description="M or F")
    chief_complaint: str
    history_of_present_illness: str
    past_medical_history: str = ""
    medications: str = ""
    social_history: str = ""
    family_history: str = ""
    vitals: Vitals = Field(default_factory=Vitals)
    physical_exam: str = ""
    labs: str = ""
    imaging: str = ""
    additional_workup: str = ""
    correct_diagnosis: str
    differential: list[str] = Field(
        default_factory=list,
        description="Reasonable differential diagnoses in order of likelihood",
    )
    explanation: str = Field(
        default="",
        description="Teaching-point explanation of why the correct diagnosis fits",
    )
    dangerous_misses: list[str] = Field(
        default_factory=list,
        description="Diagnoses that would be dangerous to miss in this presentation",
    )

    def to_prompt(self) -> str:
        """Render the vignette as a clinical prompt for an LLM."""
        sections: list[str] = []
        sections.append(
            f"A {self.patient_age}-year-old {_sex_label(self.patient_sex)} presents with "
            f"{self.chief_complaint}."
        )
        sections.append(f"\nHistory of Present Illness:\n{self.history_of_present_illness}")
        if self.past_medical_history:
            sections.append(f"\nPast Medical History:\n{self.past_medical_history}")
        if self.medications:
            sections.append(f"\nMedications:\n{self.medications}")
        if self.social_history:
            sections.append(f"\nSocial History:\n{self.social_history}")
        if self.family_history:
            sections.append(f"\nFamily History:\n{self.family_history}")
        if self.vitals.model_dump(exclude_none=True):
            sections.append(f"\nVitals:\n{_format_vitals(self.vitals)}")
        if self.physical_exam:
            sections.append(f"\nPhysical Examination:\n{self.physical_exam}")
        if self.labs:
            sections.append(f"\nLaboratory Results:\n{self.labs}")
        if self.imaging:
            sections.append(f"\nImaging:\n{self.imaging}")
        if self.additional_workup:
            sections.append(f"\nAdditional Workup:\n{self.additional_workup}")
        sections.append(
            "\n---\nBased on the clinical information above, provide:\n"
            "1. Your most likely diagnosis\n"
            "2. A ranked differential diagnosis list (up to 5)\n"
            "3. Your clinical reasoning\n"
            "4. Any critical diagnoses that must not be missed"
        )
        return "\n".join(sections)


def _sex_label(code: str) -> str:
    return "male" if code.upper() == "M" else "female"


def _format_vitals(v: Vitals) -> str:
    parts: list[str] = []
    if v.heart_rate is not None:
        parts.append(f"HR {v.heart_rate} bpm")
    if v.blood_pressure is not None:
        parts.append(f"BP {v.blood_pressure} mmHg")
    if v.respiratory_rate is not None:
        parts.append(f"RR {v.respiratory_rate}")
    if v.temperature_c is not None:
        parts.append(f"Temp {v.temperature_c} C")
    if v.spo2 is not None:
        parts.append(f"SpO2 {v.spo2}%")
    return ", ".join(parts) if parts else "Within normal limits"


# ---------------------------------------------------------------------------
# VignetteBank -- built-in collection
# ---------------------------------------------------------------------------

class VignetteBank:
    """Repository of built-in clinical vignettes for the Nidana benchmark.

    The bank ships with at least 40 expert-authored vignettes spanning
    cardiology, pulmonology, gastroenterology, nephrology, neurology,
    endocrinology, infectious disease, emergency medicine, hematology-oncology,
    rheumatology, dermatology, psychiatry, pediatrics, and OB/GYN.
    """

    def __init__(self) -> None:
        self._vignettes: list[ClinicalVignette] = _build_bank()

    # -- query helpers -------------------------------------------------------

    @property
    def all(self) -> list[ClinicalVignette]:
        return list(self._vignettes)

    def by_specialty(self, specialty: MedicalSpecialty) -> list[ClinicalVignette]:
        return [v for v in self._vignettes if v.specialty == specialty]

    def specialties_with_vignettes(self) -> list[MedicalSpecialty]:
        seen: set[MedicalSpecialty] = set()
        ordered: list[MedicalSpecialty] = []
        for v in self._vignettes:
            if v.specialty not in seen:
                seen.add(v.specialty)
                ordered.append(v.specialty)
        return ordered

    def __len__(self) -> int:
        return len(self._vignettes)


# ---------------------------------------------------------------------------
# Vignette definitions -- 40+ cases
# ---------------------------------------------------------------------------

def _build_bank() -> list[ClinicalVignette]:  # noqa: C901 (long by design)
    vs: list[ClinicalVignette] = []

    # ===== CARDIOLOGY (5) ==================================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.CARDIOLOGY,
        difficulty="moderate",
        patient_age=62,
        patient_sex="M",
        chief_complaint="substernal chest pressure radiating to the left arm for 2 hours",
        history_of_present_illness=(
            "The patient developed sudden-onset substernal chest pressure while mowing the lawn. "
            "Pain is described as 8/10, radiating to the left arm and jaw, associated with "
            "diaphoresis and nausea. He took aspirin 325 mg en route. No prior episodes."
        ),
        past_medical_history="Hypertension, hyperlipidemia, type 2 diabetes mellitus, 30-pack-year smoking history.",
        medications="Lisinopril 20 mg, atorvastatin 40 mg, metformin 1000 mg BID.",
        vitals=Vitals(heart_rate=98, blood_pressure="158/92", respiratory_rate=20, temperature_c=36.8, spo2=96),
        physical_exam="Diaphoretic, anxious. S1/S2 regular without murmurs. Lungs clear. No peripheral edema.",
        labs="Troponin I 2.4 ng/mL (ref < 0.04). BMP unremarkable. CBC normal.",
        imaging="ECG: 3-mm ST elevation in leads II, III, aVF with reciprocal ST depression in I, aVL.",
        correct_diagnosis="Acute ST-elevation myocardial infarction (inferior STEMI)",
        differential=["Unstable angina", "Aortic dissection", "Pericarditis", "Pulmonary embolism"],
        explanation="ST elevation in inferior leads with elevated troponin and classic presentation is diagnostic of inferior STEMI.",
        dangerous_misses=["Aortic dissection", "Tension pneumothorax"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.CARDIOLOGY,
        difficulty="hard",
        patient_age=34,
        patient_sex="F",
        chief_complaint="progressive exertional dyspnea and lower-extremity edema over 3 months",
        history_of_present_illness=(
            "A previously healthy woman now reports dyspnea walking one block, orthopnea requiring "
            "3 pillows, and paroxysmal nocturnal dyspnea. She delivered her first child 5 months ago. "
            "She notes bilateral leg swelling and a 10-lb weight gain over the past month."
        ),
        past_medical_history="Uncomplicated pregnancy and vaginal delivery 5 months ago. No prior cardiac history.",
        vitals=Vitals(heart_rate=110, blood_pressure="100/70", respiratory_rate=24, temperature_c=36.9, spo2=92),
        physical_exam="JVP elevated to 12 cm. Displaced PMI. S3 gallop present. Bilateral crackles at lung bases. 2+ pitting edema to mid-shins.",
        labs="BNP 1850 pg/mL. Troponin I 0.12 ng/mL. Na 131 mEq/L. Cr 1.3 mg/dL.",
        imaging="CXR: cardiomegaly with cephalization of vessels and bilateral pleural effusions. Echo: LVEF 20%, global hypokinesis, LV dilation.",
        correct_diagnosis="Peripartum cardiomyopathy",
        differential=["Dilated cardiomyopathy (idiopathic)", "Viral myocarditis", "Preeclampsia with cardiac involvement", "Thyrotoxic cardiomyopathy"],
        explanation="New-onset heart failure with severely reduced EF within months of delivery, without other identifiable cause, defines peripartum cardiomyopathy.",
        dangerous_misses=["Pulmonary embolism", "Acute myocarditis"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.CARDIOLOGY,
        difficulty="moderate",
        patient_age=72,
        patient_sex="M",
        chief_complaint="recurrent syncope over the past two weeks",
        history_of_present_illness=(
            "The patient reports three episodes of sudden loss of consciousness without prodrome, "
            "each lasting seconds, with spontaneous recovery. Episodes occur during both rest and "
            "activity. He denies seizure-like activity, incontinence, or postictal confusion."
        ),
        past_medical_history="Prior anterior MI 4 years ago, EF 35%. Hypertension.",
        medications="Carvedilol 25 mg BID, lisinopril 10 mg, aspirin 81 mg.",
        vitals=Vitals(heart_rate=42, blood_pressure="98/60", respiratory_rate=16, temperature_c=36.6, spo2=97),
        physical_exam="Bradycardic, otherwise unremarkable cardiovascular exam. No carotid bruits. Neurological exam normal.",
        labs="BMP normal. Troponin negative.",
        imaging="ECG: complete heart block with ventricular escape rate 38 bpm. No acute ST changes.",
        correct_diagnosis="Complete (third-degree) atrioventricular block",
        differential=["Sick sinus syndrome", "Vasovagal syncope", "Ventricular tachycardia with self-termination", "Orthostatic hypotension"],
        explanation="Symptomatic bradycardia with ECG showing AV dissociation and ventricular escape rhythm confirms complete heart block.",
        dangerous_misses=["Ventricular tachycardia"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.CARDIOLOGY,
        difficulty="hard",
        patient_age=28,
        patient_sex="M",
        chief_complaint="sudden cardiac arrest during a basketball game",
        history_of_present_illness=(
            "A college athlete collapsed on the court without warning. Bystanders initiated CPR "
            "and an AED delivered one shock, restoring sinus rhythm. In the ED, he is alert and "
            "oriented, reporting brief palpitations before the event. No prior similar episodes."
        ),
        past_medical_history="No known medical problems. Cousin died suddenly at age 22.",
        vitals=Vitals(heart_rate=78, blood_pressure="118/72", respiratory_rate=16, temperature_c=36.7, spo2=99),
        physical_exam="Harsh crescendo-decrescendo systolic murmur at left sternal border, increases with Valsalva and standing, decreases with squatting.",
        labs="Troponin I mildly elevated at 0.08 ng/mL. BMP normal.",
        imaging="Echo: asymmetric septal hypertrophy (22 mm), systolic anterior motion of the mitral valve, dynamic LVOT gradient 55 mmHg.",
        correct_diagnosis="Hypertrophic cardiomyopathy (HCM) with dynamic left ventricular outflow tract obstruction",
        differential=["Arrhythmogenic right ventricular cardiomyopathy", "Long QT syndrome", "Wolff-Parkinson-White syndrome", "Commotio cordis"],
        explanation="Asymmetric septal hypertrophy with SAM and dynamic LVOT obstruction on echo, plus a classic murmur that increases with Valsalva, is pathognomonic for obstructive HCM.",
        dangerous_misses=["Coronary artery anomaly", "Arrhythmogenic right ventricular cardiomyopathy"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.CARDIOLOGY,
        difficulty="moderate",
        patient_age=55,
        patient_sex="F",
        chief_complaint="acute-onset tearing chest pain radiating to the back",
        history_of_present_illness=(
            "While lifting a heavy box, the patient experienced sudden-onset 10/10 tearing chest "
            "pain radiating to the interscapular region. She feels short of breath and lightheaded. "
            "The pain has not improved with nitroglycerin."
        ),
        past_medical_history="Poorly controlled hypertension for 15 years. Marfanoid habitus noted on prior records.",
        medications="Amlodipine 10 mg (poorly adherent).",
        vitals=Vitals(heart_rate=112, blood_pressure="182/98", respiratory_rate=22, temperature_c=36.5, spo2=95),
        physical_exam="Blood pressure discrepancy: right arm 182/98, left arm 148/82. Early diastolic murmur at the right sternal border. Pulses diminished in the left upper extremity.",
        labs="D-dimer > 5000 ng/mL. Troponin I 0.06 ng/mL. Cr 1.1 mg/dL.",
        imaging="CTA chest: intimal flap extending from the ascending aorta into the descending aorta, consistent with Stanford type A dissection.",
        correct_diagnosis="Acute Stanford type A aortic dissection",
        differential=["Acute myocardial infarction", "Pulmonary embolism", "Pericarditis", "Esophageal rupture (Boerhaave syndrome)"],
        explanation="Tearing pain with BP discrepancy, aortic regurgitation murmur, and CTA showing an intimal flap originating in the ascending aorta confirms type A dissection.",
        dangerous_misses=["Acute MI from coronary malperfusion", "Cardiac tamponade"],
    ))

    # ===== PULMONOLOGY (4) =================================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.PULMONOLOGY,
        difficulty="moderate",
        patient_age=68,
        patient_sex="F",
        chief_complaint="sudden-onset dyspnea and pleuritic chest pain",
        history_of_present_illness=(
            "Three days after right total knee arthroplasty, the patient developed acute shortness "
            "of breath and sharp right-sided chest pain worsened by inspiration. She reports mild "
            "right calf swelling and tenderness."
        ),
        past_medical_history="Osteoarthritis. BMI 34. No prior VTE.",
        vitals=Vitals(heart_rate=118, blood_pressure="132/84", respiratory_rate=28, temperature_c=37.4, spo2=89),
        physical_exam="Tachypneic, using accessory muscles. Right calf is swollen, erythematous, tender. Lungs: diminished breath sounds at right base.",
        labs="D-dimer 4200 ng/mL. ABG on room air: pH 7.48, PaCO2 28 mmHg, PaO2 58 mmHg. Troponin I 0.15 ng/mL. BNP 320 pg/mL.",
        imaging="CTA chest: saddle pulmonary embolus extending into bilateral main pulmonary arteries. RV/LV ratio 1.4.",
        correct_diagnosis="Acute massive (saddle) pulmonary embolism",
        differential=["Pneumonia", "Pneumothorax", "Fat embolism syndrome", "Acute heart failure exacerbation"],
        explanation="Post-surgical immobilization, DVT signs, hypoxia with respiratory alkalosis, elevated D-dimer, and CTA confirmation of saddle PE with RV strain.",
        dangerous_misses=["Fat embolism syndrome"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.PULMONOLOGY,
        difficulty="hard",
        patient_age=45,
        patient_sex="M",
        chief_complaint="progressive dyspnea on exertion and dry cough over 6 months",
        history_of_present_illness=(
            "The patient, a sandblaster by trade for 18 years, reports gradually worsening "
            "exercise tolerance. He can now walk only 1 block before stopping. He has a persistent "
            "dry cough but denies hemoptysis, fevers, or weight loss."
        ),
        social_history="Worked in shipyard sandblasting for 18 years with inconsistent use of respiratory protection. Never smoker.",
        vitals=Vitals(heart_rate=88, blood_pressure="128/78", respiratory_rate=20, temperature_c=36.7, spo2=93),
        physical_exam="Bibasilar fine inspiratory crackles. Digital clubbing present. No wheezing.",
        labs="CBC normal. ANA negative. RF negative.",
        imaging="CXR: bilateral upper-lobe predominant nodular opacities with eggshell calcification of hilar lymph nodes. HRCT: diffuse small rounded opacities in upper lobes with progressive massive fibrosis.",
        additional_workup="PFTs: FVC 58% predicted, FEV1 62% predicted, FEV1/FVC ratio 0.82, DLCO 45% predicted. Restrictive pattern with reduced diffusing capacity.",
        correct_diagnosis="Silicosis (chronic complicated silicosis with progressive massive fibrosis)",
        differential=["Idiopathic pulmonary fibrosis", "Sarcoidosis", "Coal workers' pneumoconiosis", "Tuberculosis"],
        explanation="Extensive silica exposure, upper-lobe predominant nodules, eggshell calcification of hilar nodes, and restrictive PFTs with reduced DLCO are classic for complicated silicosis.",
        dangerous_misses=["Tuberculosis (high co-prevalence with silicosis)", "Lung malignancy"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.PULMONOLOGY,
        difficulty="moderate",
        patient_age=58,
        patient_sex="M",
        chief_complaint="worsening dyspnea, productive cough with greenish sputum, and wheezing",
        history_of_present_illness=(
            "The patient with known COPD (GOLD stage III) developed increased sputum production, "
            "change in sputum color to green, and worsening dyspnea over 3 days. He has used his "
            "albuterol inhaler 8 times today without relief."
        ),
        past_medical_history="COPD (FEV1 38% predicted). 45-pack-year smoking history, quit 2 years ago. Two prior exacerbations requiring hospitalization.",
        medications="Tiotropium, fluticasone/salmeterol, albuterol PRN.",
        vitals=Vitals(heart_rate=105, blood_pressure="142/88", respiratory_rate=28, temperature_c=37.8, spo2=86),
        physical_exam="Barrel chest, pursed-lip breathing, prolonged expiratory phase, diffuse bilateral expiratory wheezes, diminished breath sounds. Using accessory muscles.",
        labs="WBC 13,200 with left shift. ABG on 2L NC: pH 7.32, PaCO2 58 mmHg, PaO2 52 mmHg, HCO3 30 mEq/L. BMP: HCO3 30 mEq/L.",
        imaging="CXR: hyperinflated lungs, flattened diaphragms, no focal consolidation or pneumothorax.",
        correct_diagnosis="Acute exacerbation of COPD (AECOPD) with acute-on-chronic hypercapnic respiratory failure",
        differential=["Community-acquired pneumonia", "Congestive heart failure exacerbation", "Pneumothorax", "Pulmonary embolism"],
        explanation="Increased dyspnea, purulent sputum, and worsening obstruction in a known COPD patient with respiratory acidosis superimposed on metabolic compensation (elevated baseline HCO3).",
        dangerous_misses=["Pneumothorax", "Pulmonary embolism"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.PULMONOLOGY,
        difficulty="hard",
        patient_age=35,
        patient_sex="F",
        chief_complaint="recurrent spontaneous pneumothorax",
        history_of_present_illness=(
            "The patient presents with acute right-sided chest pain and dyspnea. This is her third "
            "spontaneous pneumothorax in 2 years. She also reports progressive exertional dyspnea "
            "and was recently found to have multiple renal cysts on ultrasound."
        ),
        past_medical_history="Two prior right pneumothoraces. Multiple renal cysts. Skin tags on face and neck.",
        family_history="Mother had bilateral renal tumors at age 50. Uncle had recurrent pneumothoraces.",
        vitals=Vitals(heart_rate=100, blood_pressure="122/76", respiratory_rate=24, temperature_c=36.8, spo2=93),
        physical_exam="Absent breath sounds at right apex. Multiple fibrofolliculomas (skin-colored papules) on the nose and cheeks.",
        imaging="CXR: moderate right apical pneumothorax. CT chest: multiple bilateral thin-walled cysts predominantly in lower lobes and medial/subpleural regions. CT abdomen: bilateral renal cysts and a 2.5-cm enhancing right renal mass.",
        correct_diagnosis="Birt-Hogg-Dube syndrome",
        differential=["Lymphangioleiomyomatosis (LAM)", "Langerhans cell histiocytosis", "Primary spontaneous pneumothorax", "Marfan syndrome"],
        explanation="Triad of recurrent pneumothoraces, characteristic pulmonary cysts, fibrofolliculomas, and renal tumors in a patient with family history of renal tumors and pneumothoraces is classic for BHD syndrome (FLCN gene mutation).",
        dangerous_misses=["Renal cell carcinoma (the enhancing renal mass requires urgent workup)"],
    ))

    # ===== GASTROENTEROLOGY (4) ============================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.GASTROENTEROLOGY,
        difficulty="moderate",
        patient_age=44,
        patient_sex="F",
        chief_complaint="episodic right upper quadrant abdominal pain, jaundice, and dark urine",
        history_of_present_illness=(
            "The patient reports recurrent post-prandial RUQ pain lasting 2-4 hours, especially "
            "after fatty meals. Over the past 3 days, she has developed yellow discoloration of "
            "her eyes, dark urine, and pale stools."
        ),
        past_medical_history="Obesity (BMI 36), three prior pregnancies, oral contraceptive use.",
        vitals=Vitals(heart_rate=92, blood_pressure="136/82", respiratory_rate=18, temperature_c=37.2, spo2=98),
        physical_exam="Icteric sclerae. RUQ tenderness with positive Murphy sign. No peritoneal signs.",
        labs="Total bilirubin 6.8 mg/dL (direct 5.2), AST 180 U/L, ALT 220 U/L, ALP 410 U/L, GGT 380 U/L, lipase 45 U/L, WBC 11,000.",
        imaging="RUQ ultrasound: multiple gallstones, dilated CBD to 12 mm with a distal shadowing echogenicity consistent with choledocholithiasis. No intrahepatic ductal dilation.",
        correct_diagnosis="Choledocholithiasis with obstructive jaundice",
        differential=["Acute cholecystitis", "Cholangitis (ascending)", "Pancreatic head mass", "Mirizzi syndrome"],
        explanation="Obstructive pattern (elevated direct bilirubin, ALP, GGT) with ultrasound showing CBD stone and dilation in a patient with classic biliary colic history.",
        dangerous_misses=["Ascending cholangitis (Charcot triad not complete but monitor)", "Gallstone pancreatitis"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.GASTROENTEROLOGY,
        difficulty="moderate",
        patient_age=28,
        patient_sex="M",
        chief_complaint="bloody diarrhea, abdominal cramping, and urgency for 6 weeks",
        history_of_present_illness=(
            "The patient reports 8-10 bowel movements per day with visible blood and mucus, "
            "associated with lower abdominal cramping and rectal urgency. He has lost 12 lb. "
            "He has not traveled recently and has no sick contacts."
        ),
        past_medical_history="None.",
        vitals=Vitals(heart_rate=96, blood_pressure="118/72", respiratory_rate=16, temperature_c=37.5, spo2=99),
        physical_exam="Diffuse lower abdominal tenderness without rebound. Rectal exam: bright red blood on glove.",
        labs="Hgb 10.8 g/dL, WBC 12,400, ESR 48 mm/h, CRP 42 mg/L, albumin 3.0 g/dL. Stool C. difficile and cultures negative. Fecal calprotectin 860 mcg/g.",
        imaging="Colonoscopy: continuous mucosal erythema, friability, and superficial ulceration extending from the rectum to the splenic flexure. No skip lesions. Biopsies show crypt abscesses and chronic architectural distortion.",
        correct_diagnosis="Ulcerative colitis (left-sided, moderate-severe)",
        differential=["Crohn's colitis", "Infectious colitis", "Ischemic colitis", "Clostridioides difficile colitis"],
        explanation="Continuous mucosal inflammation starting at the rectum without skip lesions, crypt abscesses, and negative infectious workup define ulcerative colitis. Severity is moderate-severe based on frequency and systemic inflammation.",
        dangerous_misses=["Toxic megacolon", "CMV colitis superinfection"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.GASTROENTEROLOGY,
        difficulty="hard",
        patient_age=52,
        patient_sex="M",
        chief_complaint="progressive dysphagia to solids and 20-lb weight loss over 3 months",
        history_of_present_illness=(
            "The patient first noticed difficulty swallowing meat and bread 3 months ago. "
            "Dysphagia has progressed to soft foods. He reports odynophagia, early satiety, "
            "and unintentional 20-lb weight loss. He has a history of chronic GERD poorly managed."
        ),
        past_medical_history="GERD for 15 years, intermittent PPI use. Barrett esophagus diagnosed 5 years ago -- missed follow-up endoscopies.",
        social_history="40-pack-year smoking, 3 beers daily. BMI 22 (previously 28).",
        vitals=Vitals(heart_rate=84, blood_pressure="128/76", respiratory_rate=16, temperature_c=36.8, spo2=98),
        physical_exam="Cachectic. Supraclavicular lymphadenopathy on the left (Virchow node). Abdomen soft, nontender.",
        labs="Hgb 10.2 g/dL, albumin 2.8 g/dL. LFTs: ALP 180 U/L, rest normal. CEA 12 ng/mL.",
        imaging="EGD: circumferential, ulcerated, partially obstructing mass in the distal esophagus at 35 cm. Biopsies: invasive adenocarcinoma, moderately differentiated. CT chest/abdomen: distal esophageal thickening with mediastinal and celiac axis lymphadenopathy, two hepatic lesions suspicious for metastases.",
        correct_diagnosis="Esophageal adenocarcinoma arising from Barrett esophagus (stage IV)",
        differential=["Esophageal squamous cell carcinoma", "Peptic stricture", "Esophageal leiomyoma", "Achalasia"],
        explanation="Progressive solid-food dysphagia with weight loss in a patient with longstanding Barrett esophagus, distal esophageal mass with adenocarcinoma on biopsy, Virchow node, and hepatic metastases.",
        dangerous_misses=["Esophageal perforation from obstruction", "Tracheoesophageal fistula"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.GASTROENTEROLOGY,
        difficulty="moderate",
        patient_age=48,
        patient_sex="M",
        chief_complaint="hematemesis and melena",
        history_of_present_illness=(
            "The patient vomited approximately 500 mL of bright red blood followed by coffee-ground "
            "emesis. He also reports black, tarry stools for 2 days. He feels lightheaded and weak. "
            "He has a history of heavy alcohol use."
        ),
        past_medical_history="Alcohol use disorder, prior alcohol-related hepatitis. No prior endoscopy.",
        social_history="1 pint of vodka daily for 20 years.",
        vitals=Vitals(heart_rate=124, blood_pressure="88/54", respiratory_rate=22, temperature_c=36.6, spo2=96),
        physical_exam="Pale, diaphoretic. Spider angiomata on chest. Splenomegaly. Ascites present. Caput medusae. Rectal exam: melena.",
        labs="Hgb 7.2 g/dL, platelets 68,000, INR 1.8, albumin 2.4 g/dL, total bilirubin 3.8 mg/dL, AST 88, ALT 42 (AST:ALT > 2:1), Cr 1.4 mg/dL.",
        imaging="Bedside ultrasound: cirrhotic liver, splenomegaly, moderate ascites, patent portal vein with hepatofugal flow.",
        correct_diagnosis="Acute variceal hemorrhage secondary to portal hypertension from alcoholic cirrhosis",
        differential=["Peptic ulcer bleeding", "Mallory-Weiss tear", "Gastric varices", "Portal hypertensive gastropathy"],
        explanation="Hematemesis in a patient with stigmata of chronic liver disease (spider angiomata, ascites, caput medusae, thrombocytopenia, coagulopathy) points to variceal hemorrhage from portal hypertension.",
        dangerous_misses=["Hemorrhagic shock", "Hepatorenal syndrome"],
    ))

    # ===== NEPHROLOGY (4) ==================================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.NEPHROLOGY,
        difficulty="moderate",
        patient_age=6,
        patient_sex="M",
        chief_complaint="periorbital edema, dark-colored urine, and decreased urine output",
        history_of_present_illness=(
            "Two weeks after a sore throat treated with amoxicillin, the child developed puffy "
            "eyes in the morning, tea-colored urine, and reduced urine output. His mother notes "
            "mild ankle swelling and that he seems more tired than usual."
        ),
        past_medical_history="Group A streptococcal pharyngitis 2 weeks ago.",
        vitals=Vitals(heart_rate=100, blood_pressure="132/86", respiratory_rate=20, temperature_c=37.0, spo2=98),
        physical_exam="Periorbital edema, mild bilateral ankle edema. No rashes. Lungs clear.",
        labs="Urinalysis: 3+ blood, 2+ protein, RBC casts. BUN 32 mg/dL, Cr 1.4 mg/dL (elevated for age). C3 low at 28 mg/dL (ref 80-160), C4 normal. ASO titer 800 IU/mL (elevated). Albumin 3.2 g/dL.",
        correct_diagnosis="Post-streptococcal glomerulonephritis (PSGN)",
        differential=["IgA nephropathy", "Membranoproliferative glomerulonephritis", "Lupus nephritis", "Hemolytic uremic syndrome"],
        explanation="Nephritic syndrome (hematuria, hypertension, edema, RBC casts) 2 weeks after streptococcal pharyngitis with low C3, normal C4, and elevated ASO titer is classic for PSGN.",
        dangerous_misses=["Rapidly progressive glomerulonephritis"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.NEPHROLOGY,
        difficulty="hard",
        patient_age=45,
        patient_sex="F",
        chief_complaint="fatigue, foamy urine, and bilateral leg swelling for 2 months",
        history_of_present_illness=(
            "The patient noticed her urine becoming increasingly frothy over the past 2 months, "
            "along with progressive bilateral lower-extremity edema that now extends to her thighs. "
            "She reports a 15-lb weight gain and dyspnea when lying flat."
        ),
        past_medical_history="No significant history. No diabetes, no hypertension.",
        vitals=Vitals(heart_rate=78, blood_pressure="126/78", respiratory_rate=16, temperature_c=36.7, spo2=97),
        physical_exam="Periorbital edema. 3+ pitting edema bilateral lower extremities. Ascites present. No rash or joint findings.",
        labs="Albumin 1.6 g/dL, total cholesterol 380 mg/dL, LDL 260 mg/dL, triglycerides 320 mg/dL. Cr 0.9 mg/dL. 24-hr urine protein: 11.2 g. Urinalysis: 4+ protein, oval fat bodies, no RBC casts. ANA/ANCA/hepatitis panel negative. PLA2R antibody positive.",
        correct_diagnosis="Membranous nephropathy (primary, PLA2R-associated)",
        differential=["Minimal change disease", "Focal segmental glomerulosclerosis", "Diabetic nephropathy", "Amyloidosis"],
        explanation="Nephrotic syndrome (massive proteinuria >3.5 g, hypoalbuminemia, hyperlipidemia, edema) in a middle-aged adult with positive PLA2R antibody is diagnostic of primary membranous nephropathy.",
        dangerous_misses=["Renal vein thrombosis (complication of nephrotic syndrome)", "Underlying malignancy (secondary membranous)"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.NEPHROLOGY,
        difficulty="moderate",
        patient_age=65,
        patient_sex="M",
        chief_complaint="confusion, muscle weakness, and nausea",
        history_of_present_illness=(
            "The patient's family brought him in after 3 days of increasing confusion, muscle "
            "weakness, nausea, and constipation. He was recently diagnosed with a lung mass and "
            "has been coughing and losing weight."
        ),
        past_medical_history="60-pack-year smoking history. Recent 15-lb weight loss.",
        vitals=Vitals(heart_rate=68, blood_pressure="148/82", respiratory_rate=16, temperature_c=36.8, spo2=95),
        physical_exam="Lethargic but arousable. Dry mucous membranes. Decreased deep tendon reflexes. No focal neurological deficits.",
        labs="Ca 14.8 mg/dL (ionized Ca 7.2 mg/dL), PTH < 5 pg/mL (suppressed), PTHrP 12 pmol/L (elevated), Cr 2.1 mg/dL (baseline 1.0), BUN 42. CXR: 4-cm RUL mass.",
        correct_diagnosis="Humoral hypercalcemia of malignancy (PTHrP-mediated) from squamous cell lung carcinoma",
        differential=["Primary hyperparathyroidism", "Sarcoidosis-related hypercalcemia", "Vitamin D toxicity", "Multiple myeloma"],
        explanation="Severe hypercalcemia with suppressed PTH and elevated PTHrP in a smoker with lung mass indicates humoral hypercalcemia of malignancy, most commonly from squamous cell carcinoma.",
        dangerous_misses=["Hypercalcemic crisis with cardiac arrhythmia"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.NEPHROLOGY,
        difficulty="moderate",
        patient_age=58,
        patient_sex="F",
        chief_complaint="severe flank pain and hematuria",
        history_of_present_illness=(
            "Acute onset of severe left flank pain radiating to the groin. Pain is colicky, "
            "rated 10/10, associated with nausea, vomiting, and gross hematuria. She reports "
            "multiple prior episodes managed conservatively."
        ),
        past_medical_history="Recurrent calcium oxalate nephrolithiasis (4 episodes in 3 years). Primary hyperparathyroidism diagnosed but not yet treated.",
        medications="Calcium and vitamin D supplements.",
        vitals=Vitals(heart_rate=102, blood_pressure="158/92", respiratory_rate=20, temperature_c=37.1, spo2=98),
        physical_exam="Severe left CVA tenderness. Abdomen soft with mild left lower quadrant tenderness.",
        labs="Cr 1.6 mg/dL (baseline 1.0), Ca 11.4 mg/dL, PTH 128 pg/mL (elevated), phosphorus 2.2 mg/dL (low), urinalysis: large blood, pH 6.8.",
        imaging="CT abdomen without contrast: 9-mm obstructing stone at left ureterovesical junction with moderate left hydronephrosis.",
        correct_diagnosis="Obstructing ureteral calculus with acute kidney injury secondary to primary hyperparathyroidism",
        differential=["Renal colic without obstruction", "Pyelonephritis", "Renal cell carcinoma", "Abdominal aortic aneurysm"],
        explanation="Recurrent calcium stones with hypercalcemia, elevated PTH, and hypophosphatemia point to primary hyperparathyroidism as the driving etiology. Current presentation is an obstructing stone causing AKI.",
        dangerous_misses=["Infected obstructing stone (pyonephrosis)", "Urosepsis"],
    ))

    # ===== NEUROLOGY (4) ===================================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.NEUROLOGY,
        difficulty="moderate",
        patient_age=71,
        patient_sex="M",
        chief_complaint="sudden onset right-sided weakness and difficulty speaking for 1 hour",
        history_of_present_illness=(
            "While eating breakfast, the patient suddenly dropped his fork and could not lift his "
            "right arm. His wife noticed his speech was slurred and his face was drooping on the "
            "right. Symptom onset was 55 minutes ago. Last known well at 7:00 AM."
        ),
        past_medical_history="Atrial fibrillation (not on anticoagulation), hypertension, type 2 diabetes.",
        medications="Metoprolol 50 mg, metformin 1000 mg BID, aspirin 81 mg.",
        vitals=Vitals(heart_rate=88, blood_pressure="178/96", respiratory_rate=18, temperature_c=36.7, spo2=97),
        physical_exam="Right facial droop (lower face). Dysarthria. Right hemiplegia: 0/5 upper, 2/5 lower extremity. Right Babinski sign. Left gaze preference. NIHSS 16.",
        labs="Glucose 168 mg/dL, INR 1.0, platelets 220,000, Cr 1.1.",
        imaging="CT head without contrast: no hemorrhage, no early infarct signs. CT angiography: left M1 segment MCA occlusion.",
        correct_diagnosis="Acute ischemic stroke (left MCA territory) -- cardioembolic, secondary to atrial fibrillation",
        differential=["Hemorrhagic stroke", "Todd paralysis (postictal)", "Hypoglycemia", "Complex migraine with aura"],
        explanation="Acute onset of contralateral hemiplegia, facial droop, and dysarthria with left MCA occlusion on CTA in a patient with AF (a known embolic source) within the thrombolytic window.",
        dangerous_misses=["Hemorrhagic transformation (must rule out hemorrhage before thrombolysis)"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.NEUROLOGY,
        difficulty="hard",
        patient_age=32,
        patient_sex="F",
        chief_complaint="recurrent episodes of unilateral vision loss and progressive leg weakness",
        history_of_present_illness=(
            "Over the past 2 years, the patient experienced three distinct neurological episodes: "
            "(1) a 2-week episode of painful vision loss in the right eye (diagnosed as optic neuritis) "
            "18 months ago, (2) a 3-week episode of numbness and tingling in the left hand 10 months ago, "
            "and (3) progressive weakness in the right leg over 4 weeks. She also reports Lhermitte sign."
        ),
        past_medical_history="Optic neuritis 18 months ago (partially resolved).",
        vitals=Vitals(heart_rate=72, blood_pressure="116/72", respiratory_rate=14, temperature_c=36.8, spo2=99),
        physical_exam="Right relative afferent pupillary defect. Right leg 4/5 strength at hip flexion and knee extension. Hyperreflexia in right lower extremity. Right Babinski sign. Decreased vibration sense in both feet. Lhermitte sign present.",
        labs="CSF: WBC 8 (lymphocytic), protein 52 mg/dL, glucose normal. CSF oligoclonal bands present (not in serum). IgG index elevated at 0.82.",
        imaging="MRI brain: multiple periventricular T2/FLAIR hyperintense lesions perpendicular to the ventricles (Dawson fingers), one enhancing lesion. MRI cervical spine: two T2 hyperintense lesions, partial ring enhancement in one.",
        correct_diagnosis="Relapsing-remitting multiple sclerosis (RRMS)",
        differential=["Neuromyelitis optica spectrum disorder (NMOSD)", "CNS vasculitis", "Neurosarcoidosis", "Acute disseminated encephalomyelitis (ADEM)"],
        explanation="Dissemination in time (multiple episodes over 2 years) and space (optic nerve, periventricular white matter, cervical cord), oligoclonal bands in CSF, and classic MRI findings satisfy the McDonald criteria for MS.",
        dangerous_misses=["Neuromyelitis optica (requires different treatment)", "CNS lymphoma"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.NEUROLOGY,
        difficulty="moderate",
        patient_age=67,
        patient_sex="M",
        chief_complaint="progressive resting tremor, slowness of movement, and shuffling gait",
        history_of_present_illness=(
            "His wife first noticed a tremor in his left hand 18 months ago, present at rest and "
            "diminishing with purposeful movement. He has become slower in daily activities, his "
            "handwriting has gotten smaller, and he shuffles when walking. He reports constipation "
            "and anosmia predating the motor symptoms."
        ),
        past_medical_history="REM sleep behavior disorder diagnosed 3 years ago. Constipation.",
        vitals=Vitals(heart_rate=72, blood_pressure="138/84", respiratory_rate=16, temperature_c=36.7, spo2=98),
        physical_exam="Masked facies, hypophonia. Pill-rolling tremor of left hand at rest. Cogwheel rigidity left > right upper extremity. Bradykinesia on finger tapping. Shuffling, narrow-based gait with reduced arm swing. Postural instability on pull test.",
        labs="TSH normal. B12 normal. BMP normal.",
        imaging="MRI brain: age-appropriate changes, no structural lesion. DaTscan: reduced dopamine transporter uptake in bilateral posterior putamen, left > right.",
        correct_diagnosis="Parkinson disease (idiopathic)",
        differential=["Essential tremor", "Drug-induced parkinsonism", "Progressive supranuclear palsy", "Multiple system atrophy"],
        explanation="Asymmetric resting tremor, bradykinesia, rigidity, and postural instability with prodromal anosmia and REM sleep behavior disorder, supported by reduced DaT uptake, is classic for idiopathic Parkinson disease.",
        dangerous_misses=["Normal pressure hydrocephalus", "Wilson disease (in younger patients)"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.NEUROLOGY,
        difficulty="hard",
        patient_age=42,
        patient_sex="F",
        chief_complaint="thunderclap headache and neck stiffness",
        history_of_present_illness=(
            "The patient experienced sudden-onset, severe headache (10/10) reaching peak intensity "
            "within seconds while straining during a bowel movement. She describes it as 'the worst "
            "headache of my life.' She developed photophobia, nausea, vomiting, and a stiff neck."
        ),
        past_medical_history="Polycystic kidney disease (ADPKD). Mother had a subarachnoid hemorrhage at age 48.",
        vitals=Vitals(heart_rate=96, blood_pressure="172/98", respiratory_rate=20, temperature_c=37.6, spo2=97),
        physical_exam="Nuchal rigidity. Photophobia. Kernig and Brudzinski signs positive. No focal motor deficits. GCS 14 (E3V5M6).",
        labs="CBC, BMP normal.",
        imaging="CT head: diffuse subarachnoid blood in the basal cisterns and sylvian fissures (Fisher grade 3). CTA: 7-mm saccular aneurysm at the anterior communicating artery.",
        correct_diagnosis="Aneurysmal subarachnoid hemorrhage (ruptured anterior communicating artery aneurysm)",
        differential=["Sentinel headache from unruptured aneurysm", "Hypertensive intracerebral hemorrhage", "Cerebral venous sinus thrombosis", "Meningitis"],
        explanation="Thunderclap headache, meningismus, CT showing diffuse SAH, and CTA identifying an AComm aneurysm in a patient with ADPKD (a risk factor for intracranial aneurysms) confirm aneurysmal SAH.",
        dangerous_misses=["Rebleeding (highest risk in first 24 hours)", "Vasospasm/delayed cerebral ischemia"],
    ))

    # ===== ENDOCRINOLOGY (4) ===============================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.ENDOCRINOLOGY,
        difficulty="moderate",
        patient_age=38,
        patient_sex="F",
        chief_complaint="weight gain, facial rounding, and purple stretch marks",
        history_of_present_illness=(
            "Over the past year the patient has gained 30 lb concentrated in her trunk and face. "
            "She developed wide purple striae on her abdomen, easy bruising, and proximal muscle "
            "weakness. She also reports irregular menses and worsening acne."
        ),
        past_medical_history="New-onset hypertension (6 months ago), glucose intolerance.",
        vitals=Vitals(heart_rate=82, blood_pressure="156/98", respiratory_rate=16, temperature_c=36.8, spo2=98),
        physical_exam="Moon facies, dorsocervical fat pad (buffalo hump). Facial plethora. Wide (>1 cm) violaceous striae on abdomen and thighs. Proximal muscle weakness (cannot rise from chair without arms). Thin skin with ecchymoses.",
        labs="24-hr urine free cortisol: 380 mcg (ref < 50). Late-night salivary cortisol: 0.52 mcg/dL (ref < 0.09). 1-mg overnight dexamethasone suppression test: AM cortisol 18 mcg/dL (ref < 1.8). ACTH: 58 pg/mL (normal-high, indicating ACTH-dependent). High-dose dexamethasone suppression: cortisol suppresses by > 50%. MRI pituitary: 8-mm enhancing adenoma.",
        correct_diagnosis="Cushing disease (ACTH-secreting pituitary adenoma)",
        differential=["Ectopic ACTH syndrome", "Adrenal adenoma (ACTH-independent Cushing)", "Pseudo-Cushing syndrome (depression/alcoholism)", "Exogenous corticosteroid use"],
        explanation="Confirmed hypercortisolism (elevated UFC, salivary cortisol, failed low-dose dex suppression), ACTH-dependent source, suppression with high-dose dexamethasone, and pituitary adenoma on MRI identify Cushing disease.",
        dangerous_misses=["Ectopic ACTH (e.g., small-cell lung cancer if ACTH source unclear)"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.ENDOCRINOLOGY,
        difficulty="moderate",
        patient_age=25,
        patient_sex="F",
        chief_complaint="palpitations, weight loss, and heat intolerance",
        history_of_present_illness=(
            "Over 3 months the patient has lost 15 lb despite increased appetite. She reports "
            "palpitations, tremor, heat intolerance, frequent bowel movements, and menstrual "
            "irregularity. She has noticed her eyes appear more prominent."
        ),
        past_medical_history="No significant history.",
        family_history="Mother with Hashimoto thyroiditis.",
        vitals=Vitals(heart_rate=112, blood_pressure="148/62", respiratory_rate=18, temperature_c=37.2, spo2=99),
        physical_exam="Diffusely enlarged, nontender thyroid with audible bruit. Lid lag and proptosis bilateral. Fine tremor of outstretched hands. Pretibial myxedema. Brisk reflexes throughout.",
        labs="TSH < 0.01 mIU/L, free T4 4.8 ng/dL (ref 0.8-1.8), free T3 12 pg/mL (ref 2.3-4.2). TSI (thyroid-stimulating immunoglobulin) positive. Anti-TPO elevated.",
        imaging="Thyroid uptake scan: diffusely increased uptake (55%, ref 10-30%).",
        correct_diagnosis="Graves disease",
        differential=["Toxic multinodular goiter", "Toxic adenoma", "Subacute thyroiditis", "Factitious thyrotoxicosis"],
        explanation="Thyrotoxicosis with diffuse goiter, ophthalmopathy (proptosis, lid lag), pretibial myxedema, positive TSI, and diffusely increased radioiodine uptake is pathognomonic for Graves disease.",
        dangerous_misses=["Thyroid storm"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.ENDOCRINOLOGY,
        difficulty="hard",
        patient_age=34,
        patient_sex="M",
        chief_complaint="episodic headaches, palpitations, and diaphoresis with severe hypertension",
        history_of_present_illness=(
            "The patient has experienced paroxysmal episodes of severe headache, palpitations, "
            "profuse sweating, and anxiety lasting 20-30 minutes, occurring 2-3 times per week "
            "for 4 months. During one episode at urgent care, his BP was recorded at 240/140 mmHg. "
            "Between episodes, his BP is mildly elevated."
        ),
        past_medical_history="No known medical problems. Father had thyroid surgery.",
        vitals=Vitals(heart_rate=104, blood_pressure="198/112", respiratory_rate=20, temperature_c=37.0, spo2=98),
        physical_exam="Diaphoretic, anxious. Tremor. No thyroid nodules. No abdominal masses palpable.",
        labs="24-hr urine fractionated metanephrines: normetanephrine 2400 mcg (ref < 900), metanephrine 1100 mcg (ref < 400). Plasma-free metanephrines: normetanephrine 5.2 nmol/L (ref < 0.9). Serum calcitonin elevated. Serum calcium 11.2 mg/dL, PTH 95 pg/mL.",
        imaging="CT abdomen with contrast: 4.5-cm right adrenal mass with heterogeneous enhancement. MRI abdomen: T2 hyperintense right adrenal mass ('light bulb' sign). MIBG scan: avid uptake in right adrenal mass only.",
        additional_workup="Genetic testing: RET proto-oncogene mutation confirmed.",
        correct_diagnosis="Pheochromocytoma as part of Multiple Endocrine Neoplasia type 2A (MEN2A)",
        differential=["Essential hypertension", "Panic disorder", "Hyperthyroidism", "Carcinoid syndrome"],
        explanation="Classic paroxysmal triad (headache, palpitations, diaphoresis), markedly elevated metanephrines, characteristic adrenal mass on imaging, plus concurrent hyperparathyroidism and elevated calcitonin (medullary thyroid carcinoma) with confirmed RET mutation define pheochromocytoma in MEN2A.",
        dangerous_misses=["Hypertensive crisis/stroke during uncontrolled catecholamine release", "Medullary thyroid carcinoma"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.ENDOCRINOLOGY,
        difficulty="hard",
        patient_age=19,
        patient_sex="M",
        chief_complaint="polyuria, polydipsia, abdominal pain, and vomiting",
        history_of_present_illness=(
            "A previously healthy college student presents with 3 weeks of increased thirst and "
            "urination, followed by 2 days of nausea, vomiting, diffuse abdominal pain, and "
            "fruity-smelling breath. His roommate says he has been drinking liters of water and "
            "losing weight rapidly."
        ),
        past_medical_history="None. No family history of diabetes.",
        vitals=Vitals(heart_rate=128, blood_pressure="96/58", respiratory_rate=32, temperature_c=36.4, spo2=98),
        physical_exam="Kussmaul respiration. Fruity breath odor. Dry mucous membranes, poor skin turgor. Diffuse abdominal tenderness without peritoneal signs. Lethargic but oriented.",
        labs="Glucose 486 mg/dL, pH 7.12, PaCO2 14 mmHg, HCO3 6 mEq/L, anion gap 28, beta-hydroxybutyrate 8.4 mmol/L, Na 128 mEq/L (corrected 134), K 5.8 mEq/L (but total body K depleted), BUN 38, Cr 2.0, serum osmolality 312. Urinalysis: 4+ glucose, large ketones.",
        correct_diagnosis="Diabetic ketoacidosis (DKA) -- new-onset type 1 diabetes mellitus",
        differential=["Alcoholic ketoacidosis", "Starvation ketoacidosis", "Toxic ingestion (methanol, ethylene glycol)", "Acute pancreatitis"],
        explanation="Severe hyperglycemia, high anion gap metabolic acidosis, elevated beta-hydroxybutyrate, and ketonuria in a young patient with new polyuria/polydipsia is classic DKA from new-onset T1DM.",
        dangerous_misses=["Cerebral edema (especially during treatment)", "Hypokalemia during insulin therapy"],
    ))

    # ===== INFECTIOUS DISEASE (4) ==========================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.INFECTIOUS_DISEASE,
        difficulty="hard",
        patient_age=42,
        patient_sex="M",
        chief_complaint="fever, night sweats, and a new heart murmur",
        history_of_present_illness=(
            "The patient has had low-grade fevers (up to 38.8 C), drenching night sweats, fatigue, "
            "and a 10-lb weight loss over 6 weeks. He was seen for a dental extraction 8 weeks ago "
            "without antibiotic prophylaxis. He reports myalgias and a painful lesion on his finger."
        ),
        past_medical_history="Bicuspid aortic valve diagnosed incidentally on echo 5 years ago.",
        vitals=Vitals(heart_rate=96, blood_pressure="132/52", respiratory_rate=18, temperature_c=38.6, spo2=97),
        physical_exam="New 3/6 early diastolic decrescendo murmur at left sternal border (aortic regurgitation). Osler nodes on the 2nd and 4th fingertips. Janeway lesions on palms. Splinter hemorrhages in nail beds. Splenomegaly.",
        labs="WBC 14,200, Hgb 10.8, ESR 68, CRP 82. Blood cultures (3 sets): all growing Streptococcus mutans. Complement C3/C4 low. RF positive. Urinalysis: microscopic hematuria with RBC casts.",
        imaging="TTE: aortic valve vegetation 14 mm on bicuspid valve with moderate aortic regurgitation. TEE confirms mobile vegetation.",
        correct_diagnosis="Subacute bacterial endocarditis (Streptococcus mutans) on bicuspid aortic valve",
        differential=["Rheumatic fever", "Atrial myxoma", "Systemic lupus erythematosus", "Lymphoma with cardiac involvement"],
        explanation="Modified Duke criteria fulfilled: 2 major criteria (persistently positive blood cultures with typical organism, vegetation on echo with new regurgitation) plus multiple minor criteria (fever, Osler nodes, Janeway lesions, immunologic phenomena).",
        dangerous_misses=["Septic emboli to brain (mycotic aneurysm/stroke)", "Heart failure from acute valvular destruction"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.INFECTIOUS_DISEASE,
        difficulty="moderate",
        patient_age=35,
        patient_sex="M",
        chief_complaint="chronic cough, hemoptysis, night sweats, and weight loss",
        history_of_present_illness=(
            "The patient, a recent immigrant from Southeast Asia, has had a productive cough for "
            "3 months with occasional blood-streaked sputum. He reports drenching night sweats, "
            "a 20-lb weight loss, and low-grade fevers. He lives in a congregate housing setting."
        ),
        social_history="Immigrated from Cambodia 8 months ago. Lives in a shelter. No known HIV. No drug use.",
        vitals=Vitals(heart_rate=90, blood_pressure="118/72", respiratory_rate=20, temperature_c=38.2, spo2=95),
        physical_exam="Thin, cachectic. Dullness to percussion and bronchial breath sounds over right upper lobe. Cervical lymphadenopathy.",
        labs="WBC 10,800, Hgb 11.2. HIV test: negative. QuantiFERON-TB Gold: positive.",
        imaging="CXR: right upper lobe cavitary lesion (3 cm) with surrounding infiltrate, ipsilateral hilar lymphadenopathy. CT chest confirms thick-walled cavitary lesion.",
        additional_workup="Sputum AFB smear: positive for acid-fast bacilli (3+). GeneXpert MTB/RIF: Mycobacterium tuberculosis detected, rifampin resistance not detected.",
        correct_diagnosis="Pulmonary tuberculosis (cavitary, smear-positive)",
        differential=["Lung abscess (bacterial)", "Lung cancer (cavitary)", "Non-tuberculous mycobacterial infection", "Pulmonary aspergillosis"],
        explanation="Classic reactivation TB: upper-lobe cavitary lesion, constitutional symptoms, positive AFB smear, and GeneXpert confirmation in an epidemiologically at-risk patient.",
        dangerous_misses=["MDR-TB (must await full susceptibility)", "Miliary dissemination"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.INFECTIOUS_DISEASE,
        difficulty="moderate",
        patient_age=29,
        patient_sex="F",
        chief_complaint="high fever, severe headache, myalgias, and rash after travel",
        history_of_present_illness=(
            "The patient returned from a 2-week trip to sub-Saharan Africa 10 days ago. She "
            "developed high fevers (up to 40.2 C) with rigors cycling every 48 hours, drenching "
            "sweats, severe headache, myalgias, and malaise. She took no malaria prophylaxis."
        ),
        past_medical_history="None.",
        vitals=Vitals(heart_rate=116, blood_pressure="98/62", respiratory_rate=24, temperature_c=40.1, spo2=94),
        physical_exam="Acutely ill, rigors. Jaundice. Hepatosplenomegaly. No meningeal signs. No rash.",
        labs="WBC 3,800, Hgb 9.4, platelets 42,000, total bilirubin 4.2 (indirect 3.4), LDH 580, haptoglobin < 10, reticulocytes 8%, Cr 2.1, glucose 62 mg/dL, lactate 4.8 mmol/L.",
        additional_workup="Peripheral blood smear: ring-form trophozoites within RBCs, some with multiple ring forms per cell and banana-shaped gametocytes. Parasitemia 8%. Rapid diagnostic test (HRP2): positive.",
        correct_diagnosis="Severe Plasmodium falciparum malaria",
        differential=["Dengue fever", "Typhoid fever", "Leptospirosis", "Viral hemorrhagic fever"],
        explanation="Cyclic fevers after travel to endemic region without prophylaxis, hemolytic anemia, thrombocytopenia, parasitemia 8% with P. falciparum morphology (multiply-infected RBCs, banana gametocytes), hypoglycemia, and end-organ dysfunction define severe falciparum malaria.",
        dangerous_misses=["Cerebral malaria", "Severe anemia with hemodynamic compromise"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.INFECTIOUS_DISEASE,
        difficulty="hard",
        patient_age=55,
        patient_sex="M",
        chief_complaint="progressive confusion, fever, and seizure",
        history_of_present_illness=(
            "The patient was found confused by his partner, with 3 days of fever, headache, and "
            "behavioral changes. Today he had a witnessed generalized tonic-clonic seizure lasting "
            "2 minutes. His partner reports the patient has been 'acting strange' -- saying bizarre "
            "things and being unusually agitated."
        ),
        past_medical_history="No significant history. No immunocompromise.",
        vitals=Vitals(heart_rate=108, blood_pressure="142/88", respiratory_rate=20, temperature_c=39.2, spo2=96),
        physical_exam="GCS 11 (E3V3M5). Disoriented, intermittently agitated. Nuchal rigidity mild. Temporal lobe signs: olfactory hallucinations reported before seizure. No focal motor deficits.",
        labs="WBC 12,400, BMP normal, LFTs normal.",
        imaging="MRI brain: asymmetric T2/FLAIR hyperintensity in bilateral medial temporal lobes (left > right) and insular cortex with restricted diffusion. No ring enhancement.",
        additional_workup="LP: opening pressure 22 cmH2O, WBC 84 (95% lymphocytes), RBC 120, protein 78 mg/dL, glucose 62 mg/dL (serum 108). CSF HSV PCR: positive.",
        correct_diagnosis="Herpes simplex encephalitis (HSV-1)",
        differential=["Autoimmune (anti-NMDA receptor) encephalitis", "Bacterial meningitis", "CNS lymphoma", "Status epilepticus (other cause)"],
        explanation="Acute encephalitis with temporal lobe predilection on MRI, lymphocytic pleocytosis with RBCs in CSF (hemorrhagic component), and positive HSV PCR confirm HSV encephalitis.",
        dangerous_misses=["Delay in acyclovir initiation (must treat empirically before PCR returns)", "Cerebral edema with herniation"],
    ))

    # ===== EMERGENCY MEDICINE (3) ==========================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.EMERGENCY_MEDICINE,
        difficulty="moderate",
        patient_age=22,
        patient_sex="M",
        chief_complaint="severe epigastric pain radiating to the back, nausea, and vomiting after binge drinking",
        history_of_present_illness=(
            "The patient drank 12+ beers at a party last night. He woke up with severe, constant "
            "epigastric pain radiating straight through to the back, rated 9/10. Pain is worse "
            "when lying flat and improved when leaning forward. He has been vomiting repeatedly."
        ),
        social_history="Heavy binge drinker on weekends. No drug use.",
        vitals=Vitals(heart_rate=118, blood_pressure="102/64", respiratory_rate=22, temperature_c=37.8, spo2=97),
        physical_exam="Distressed, lying in fetal position. Epigastric tenderness with voluntary guarding. Diminished bowel sounds. No Grey Turner or Cullen signs.",
        labs="Lipase 2,840 U/L (ref < 60), amylase 1,200 U/L, WBC 16,800, Hgb 14.8, BUN 28, Cr 1.3, Ca 8.2, glucose 198, triglycerides 180, ALT 42, AST 68, total bilirubin 1.4.",
        imaging="CT abdomen with contrast: diffuse pancreatic edema with peripancreatic fat stranding and small peripancreatic fluid collection. No necrosis. No gallstones.",
        correct_diagnosis="Acute alcohol-induced pancreatitis (interstitial edematous, moderately severe)",
        differential=["Peptic ulcer disease/perforation", "Acute cholecystitis", "Mesenteric ischemia", "Abdominal aortic aneurysm rupture"],
        explanation="Epigastric pain radiating to the back with lipase >3x ULN after heavy alcohol use, confirmed by CT showing pancreatic edema without necrosis, meets criteria for acute pancreatitis (alcohol etiology).",
        dangerous_misses=["Necrotizing pancreatitis", "Perforated peptic ulcer"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.EMERGENCY_MEDICINE,
        difficulty="hard",
        patient_age=4,
        patient_sex="M",
        chief_complaint="high fever, irritability, and petechial rash progressing rapidly",
        history_of_present_illness=(
            "The child was well yesterday morning but developed fever to 39.8 C by the afternoon, "
            "becoming increasingly irritable and lethargic. Over the past 4 hours, parents noticed "
            "a spreading rash starting as small red spots on the legs, now covering the trunk and "
            "extremities, some lesions are not blanching."
        ),
        past_medical_history="Up to date on vaccinations except missed MenACWY booster. Born full-term, healthy.",
        vitals=Vitals(heart_rate=168, blood_pressure="72/40", respiratory_rate=36, temperature_c=39.6, spo2=94),
        physical_exam="Lethargic, poor eye contact. Bulging fontanelle not applicable (age 4). Nuchal rigidity difficult to assess due to irritability. Widespread non-blanching petechiae and purpura, some coalescent. CRT > 4 seconds. Cool extremities.",
        labs="WBC 22,400 with 85% PMNs and 12% bands, Hgb 11.2, platelets 48,000, CRP 280 mg/L, procalcitonin 42 ng/mL, lactate 5.8 mmol/L, INR 2.1, fibrinogen 90 mg/dL, D-dimer > 10,000.",
        imaging="CXR: no focal consolidation.",
        additional_workup="LP deferred due to hemodynamic instability and coagulopathy. Blood cultures obtained prior to antibiotics.",
        correct_diagnosis="Meningococcemia (Neisseria meningitidis septicemia) with purpura fulminans and DIC",
        differential=["Rocky Mountain spotted fever", "Henoch-Schonlein purpura with sepsis", "Staphylococcal toxic shock syndrome", "Acute leukemia"],
        explanation="Fulminant presentation with non-blanching petechiae/purpura, septic shock, DIC (thrombocytopenia, elevated INR, low fibrinogen, elevated D-dimer), and rapid progression in a child with incomplete meningococcal vaccination.",
        dangerous_misses=["Adrenal hemorrhage (Waterhouse-Friderichsen syndrome)", "Compartment syndrome from purpura fulminans"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.EMERGENCY_MEDICINE,
        difficulty="moderate",
        patient_age=45,
        patient_sex="F",
        chief_complaint="anaphylaxis after a bee sting",
        history_of_present_illness=(
            "The patient was stung by a bee 20 minutes ago. She immediately developed generalized "
            "urticaria, facial and lip swelling, throat tightness with stridor, and lightheadedness. "
            "She used her expired EpiPen but symptoms are worsening."
        ),
        past_medical_history="Prior bee sting allergy (localized reaction only). Asthma.",
        medications="Fluticasone/salmeterol inhaler.",
        vitals=Vitals(heart_rate=136, blood_pressure="76/42", respiratory_rate=28, temperature_c=36.8, spo2=88),
        physical_exam="Marked angioedema of face, lips, and tongue. Audible stridor and diffuse wheezing. Diffuse urticaria over trunk and extremities. Altered mental status (drowsy).",
        correct_diagnosis="Anaphylactic shock (Hymenoptera venom-induced anaphylaxis)",
        differential=["Severe allergic reaction without anaphylaxis", "Hereditary angioedema", "ACE-inhibitor angioedema", "Vocal cord dysfunction"],
        explanation="Acute multisystem allergic reaction after known allergen exposure: skin (urticaria, angioedema), respiratory (stridor, wheezing), cardiovascular (hypotension, tachycardia) meeting anaphylaxis criteria with shock.",
        dangerous_misses=["Complete airway obstruction", "Biphasic anaphylaxis (recurrence hours later)"],
    ))

    # ===== HEMATOLOGY-ONCOLOGY (3) =========================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.HEMATOLOGY_ONCOLOGY,
        difficulty="moderate",
        patient_age=22,
        patient_sex="M",
        chief_complaint="fatigue, easy bruising, gingival bleeding, and recurrent fevers for 3 weeks",
        history_of_present_illness=(
            "The patient has experienced progressive fatigue, easy bruising, bleeding from his "
            "gums when brushing teeth, and fevers up to 38.9 C over 3 weeks. He reports dyspnea "
            "on exertion and petechiae on his legs."
        ),
        past_medical_history="Previously healthy.",
        vitals=Vitals(heart_rate=108, blood_pressure="112/68", respiratory_rate=20, temperature_c=38.4, spo2=97),
        physical_exam="Pallor, petechiae on lower extremities, ecchymoses on arms. Gingival hypertrophy with bleeding. Hepatosplenomegaly. No lymphadenopathy.",
        labs="WBC 68,000 with 82% blasts, Hgb 6.8 g/dL, platelets 18,000, LDH 1,200, uric acid 10.4 mg/dL. Peripheral smear: numerous blasts with Auer rods, some with multiple Auer rods in a 'faggot' pattern. DIC panel: PT 18s, fibrinogen 98, D-dimer elevated.",
        additional_workup="Flow cytometry: blasts positive for CD13, CD33, MPO. Cytogenetics: t(15;17)(q24;q21). FISH: PML-RARA fusion positive.",
        correct_diagnosis="Acute promyelocytic leukemia (APL, AML-M3) with DIC",
        differential=["Acute myeloid leukemia (other subtypes)", "Acute lymphoblastic leukemia", "Myelodysplastic syndrome", "Aplastic anemia"],
        explanation="Leukocytosis with blasts containing Auer rods in faggot cells, gingival hypertrophy, DIC, and t(15;17) with PML-RARA fusion is diagnostic of APL. This is a hematologic emergency requiring immediate ATRA therapy.",
        dangerous_misses=["Fatal hemorrhage from DIC (requires urgent ATRA)", "Tumor lysis syndrome"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.HEMATOLOGY_ONCOLOGY,
        difficulty="moderate",
        patient_age=68,
        patient_sex="F",
        chief_complaint="bone pain, fatigue, and recurrent infections",
        history_of_present_illness=(
            "Over 4 months, the patient has developed progressive back pain, fatigue, and two "
            "episodes of pneumonia. She reports unintentional 12-lb weight loss. Recently she "
            "noticed foamy urine."
        ),
        past_medical_history="Osteopenia.",
        vitals=Vitals(heart_rate=88, blood_pressure="132/78", respiratory_rate=16, temperature_c=37.0, spo2=96),
        physical_exam="Pallor. Point tenderness over lumbar spine and ribs. No lymphadenopathy or hepatosplenomegaly.",
        labs="Hgb 8.4 g/dL, WBC 4,200 (normal differential), platelets 142,000. Cr 2.4 mg/dL, Ca 12.1 mg/dL, total protein 11.2 g/dL, albumin 3.2 g/dL (globulin gap 8.0). ESR 120 mm/hr. 24-hr urine protein 4.2 g with Bence Jones proteinuria. SPEP: M-spike 4.8 g/dL (IgG kappa). Serum free light chains: kappa 480 mg/L, lambda 12 mg/L.",
        imaging="Skeletal survey: multiple lytic (punched-out) lesions in skull, ribs, pelvis, and lumbar spine. Compression fracture L3.",
        additional_workup="Bone marrow biopsy: 62% clonal plasma cells.",
        correct_diagnosis="Multiple myeloma (IgG kappa, stage III)",
        differential=["Metastatic carcinoma to bone", "Waldenstrom macroglobulinemia", "Primary amyloidosis", "Monoclonal gammopathy of undetermined significance (MGUS)"],
        explanation="CRAB criteria fulfilled (Calcium elevated, Renal insufficiency, Anemia, Bone lesions) with M-spike, Bence Jones proteinuria, and >10% clonal plasma cells on marrow biopsy confirm multiple myeloma.",
        dangerous_misses=["Spinal cord compression from vertebral fracture", "Hyperviscosity syndrome"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.HEMATOLOGY_ONCOLOGY,
        difficulty="moderate",
        patient_age=30,
        patient_sex="M",
        chief_complaint="painless neck swelling, night sweats, and pruritus",
        history_of_present_illness=(
            "The patient noticed a painless lump on the left side of his neck 2 months ago that "
            "has gradually enlarged. He reports drenching night sweats, unintentional 15-lb weight "
            "loss over 2 months, and generalized pruritus worse after hot showers. He also describes "
            "pain in the neck mass after drinking alcohol."
        ),
        vitals=Vitals(heart_rate=78, blood_pressure="122/74", respiratory_rate=14, temperature_c=37.0, spo2=99),
        physical_exam="3-cm rubbery, nontender, mobile lymph node in left anterior cervical chain. Additional 2-cm node in left axilla. No hepatosplenomegaly.",
        labs="CBC: WBC 11,200 with mild eosinophilia, Hgb 12.8, platelets 380,000. ESR 48, LDH 280. Albumin 3.4.",
        imaging="CT chest/abdomen/pelvis: bilateral cervical, mediastinal, and left axillary lymphadenopathy. Largest mediastinal node 4 cm. No hepatosplenomegaly.",
        additional_workup="Excisional biopsy of cervical node: Reed-Sternberg cells in a background of lymphocytes, eosinophils, and histiocytes. Immunohistochemistry: CD15+, CD30+. PET scan: avid uptake in cervical, mediastinal, and axillary nodes (stage IIB).",
        correct_diagnosis="Classic Hodgkin lymphoma (nodular sclerosis subtype, stage IIB)",
        differential=["Non-Hodgkin lymphoma", "Sarcoidosis", "Tuberculosis (lymph node)", "Reactive lymphadenopathy"],
        explanation="B symptoms (night sweats, weight loss), painless lymphadenopathy, alcohol-related lymph node pain (pathognomonic for Hodgkin), Reed-Sternberg cells with CD15/CD30 positivity confirm classic Hodgkin lymphoma.",
        dangerous_misses=["Superior vena cava syndrome (large mediastinal mass)"],
    ))

    # ===== RHEUMATOLOGY (3) ================================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.RHEUMATOLOGY,
        difficulty="moderate",
        patient_age=32,
        patient_sex="F",
        chief_complaint="joint pain, facial rash, and fatigue",
        history_of_present_illness=(
            "Over 4 months, the patient has developed symmetric joint pain and swelling in her "
            "hands (MCPs and PIPs), wrists, and knees. She noticed a rash across her cheeks and "
            "nose that worsens with sun exposure. She reports fatigue, oral ulcers, hair loss, "
            "and Raynaud phenomenon."
        ),
        past_medical_history="Two first-trimester miscarriages.",
        vitals=Vitals(heart_rate=82, blood_pressure="138/88", respiratory_rate=16, temperature_c=37.4, spo2=98),
        physical_exam="Malar (butterfly) rash sparing the nasolabial folds. Oral ulcers on hard palate. Synovitis in bilateral MCPs, PIPs, and wrists. Diffuse non-scarring alopecia.",
        labs="ANA 1:640 (homogeneous), anti-dsDNA 1:320 (positive), anti-Smith antibody positive. C3 48 mg/dL (low), C4 8 mg/dL (low). CBC: WBC 3,200 (lymphopenia 800), Hgb 10.4, platelets 108,000. ESR 62. Urinalysis: 2+ protein, 15 RBCs/hpf, RBC casts. 24-hr urine protein 2.1 g. Cr 1.1.",
        correct_diagnosis="Systemic lupus erythematosus (SLE) with lupus nephritis",
        differential=["Mixed connective tissue disease", "Rheumatoid arthritis", "Drug-induced lupus", "Antiphospholipid syndrome (primary)"],
        explanation="Meets multiple ACR/SLICC criteria: malar rash, oral ulcers, arthritis, proteinuria with active sediment, lymphopenia, thrombocytopenia, positive ANA, anti-dsDNA, anti-Smith, and low complement. Active nephritis requires biopsy for classification.",
        dangerous_misses=["Rapidly progressive lupus nephritis (class IV)", "Catastrophic antiphospholipid syndrome"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.RHEUMATOLOGY,
        difficulty="hard",
        patient_age=58,
        patient_sex="F",
        chief_complaint="progressive proximal muscle weakness and skin rash",
        history_of_present_illness=(
            "Over 3 months, the patient has developed progressive weakness of proximal muscles, "
            "making it difficult to climb stairs, rise from a chair, or lift her arms above her "
            "head. She developed a violaceous rash around her eyes and over her knuckles."
        ),
        past_medical_history="No significant history.",
        vitals=Vitals(heart_rate=80, blood_pressure="128/78", respiratory_rate=16, temperature_c=36.8, spo2=97),
        physical_exam="Heliotrope rash (violaceous discoloration of upper eyelids with periorbital edema). Gottron papules over MCPs and PIPs. V-sign rash on anterior chest. Symmetric proximal muscle weakness: deltoids 3/5, hip flexors 3/5. No distal weakness.",
        labs="CK 4,800 U/L (ref < 200), aldolase 18 U/L (elevated), AST 120, ALT 88, LDH 680. ANA 1:160. Anti-Mi-2 antibody positive. Anti-Jo-1 negative. ESR 42.",
        imaging="MRI thigh: diffuse symmetric edema in proximal musculature on STIR sequences. CT chest/abdomen/pelvis: 3-cm ovarian mass.",
        additional_workup="EMG: myopathic pattern (short, small, polyphasic motor unit potentials) with fibrillation potentials. Muscle biopsy: perifascicular atrophy with perivascular inflammation.",
        correct_diagnosis="Dermatomyositis with occult ovarian malignancy (cancer-associated myositis)",
        differential=["Polymyositis", "Inclusion body myositis", "Statin-induced myopathy", "Hypothyroid myopathy"],
        explanation="Pathognomonic skin findings (heliotrope rash, Gottron papules), proximal weakness, elevated CK, myopathic EMG, perifascicular atrophy on biopsy define dermatomyositis. Age >40 onset mandates malignancy screening -- ovarian mass found.",
        dangerous_misses=["Interstitial lung disease (if anti-MDA5 positive)", "Metastatic ovarian cancer"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.RHEUMATOLOGY,
        difficulty="moderate",
        patient_age=48,
        patient_sex="M",
        chief_complaint="acute severe pain and swelling of the right first metatarsophalangeal joint",
        history_of_present_illness=(
            "The patient woke at 3 AM with excruciating pain in his right great toe, rated 10/10. "
            "The joint is red, hot, and so tender that he cannot bear the weight of a bedsheet on "
            "it. He ate a large steak dinner and drank a bottle of wine last night. This is his "
            "third similar episode in 2 years."
        ),
        past_medical_history="Hypertension, chronic kidney disease stage 3, obesity (BMI 34).",
        medications="Hydrochlorothiazide 25 mg, lisinopril 20 mg.",
        vitals=Vitals(heart_rate=92, blood_pressure="148/92", respiratory_rate=16, temperature_c=37.8, spo2=98),
        physical_exam="Right 1st MTP joint: exquisitely tender, erythematous, warm, swollen. Tophi visible on bilateral ear helices and left olecranon bursa.",
        labs="Serum uric acid 11.2 mg/dL. WBC 13,400, CRP 68. Cr 1.8 mg/dL. Synovial fluid: WBC 48,000 (90% PMNs), negatively birefringent needle-shaped crystals under polarized microscopy. Gram stain and culture negative.",
        correct_diagnosis="Acute gouty arthritis (podagra) with chronic tophaceous gout",
        differential=["Septic arthritis", "Pseudogout (CPPD)", "Cellulitis", "Reactive arthritis"],
        explanation="Acute monoarthritis of the 1st MTP with negatively birefringent needle-shaped monosodium urate crystals on synovial fluid analysis is diagnostic of gout. Tophi indicate chronic disease. HCTZ is a contributing factor.",
        dangerous_misses=["Septic arthritis (can coexist with gout -- always send cultures)"],
    ))

    # ===== DERMATOLOGY (3) =================================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.DERMATOLOGY,
        difficulty="moderate",
        patient_age=72,
        patient_sex="M",
        chief_complaint="non-healing ulcerated nodule on the nose for 6 months",
        history_of_present_illness=(
            "The patient noticed a small pearly bump on the left nasal ala 6 months ago. It has "
            "slowly enlarged and now has a central ulcer with rolled borders that bleeds with "
            "minor trauma. It does not hurt."
        ),
        past_medical_history="History of extensive sun exposure (farmer for 50 years). Fair skin, blue eyes. Prior actinic keratoses treated with cryotherapy.",
        vitals=Vitals(heart_rate=72, blood_pressure="138/82", respiratory_rate=14, temperature_c=36.7, spo2=98),
        physical_exam="0.8-cm nodule on left nasal ala with pearly, translucent rolled borders, central ulceration with crusting, and prominent telangiectasias. No palpable cervical or preauricular lymphadenopathy.",
        labs="None indicated at this stage.",
        imaging="Dermoscopy: arborizing vessels, blue-gray ovoid nests, leaf-like structures.",
        additional_workup="Shave biopsy: nests of basaloid cells with peripheral palisading and retraction artifact, extending from the epidermis into the dermis.",
        correct_diagnosis="Basal cell carcinoma (nodular type)",
        differential=["Squamous cell carcinoma", "Keratoacanthoma", "Amelanotic melanoma", "Sebaceous hyperplasia"],
        explanation="Classic presentation: pearly nodule with telangiectasias and rolled borders on sun-exposed skin in a fair-skinned elderly patient. Biopsy confirms basaloid nests with peripheral palisading.",
        dangerous_misses=["Amelanotic melanoma (must biopsy to distinguish)", "Perineural invasion (risk given nasal location)"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.DERMATOLOGY,
        difficulty="hard",
        patient_age=65,
        patient_sex="F",
        chief_complaint="widespread tense blisters on the trunk and extremities",
        history_of_present_illness=(
            "Over 3 weeks, the patient developed intensely pruritic urticarial plaques that "
            "progressed to large, tense blisters on her abdomen, thighs, and arms. The blisters "
            "are clear-fluid-filled and do not easily rupture. She has no mucosal involvement."
        ),
        past_medical_history="Hypertension, type 2 diabetes, recently started linagliptin 2 months ago.",
        vitals=Vitals(heart_rate=78, blood_pressure="142/84", respiratory_rate=16, temperature_c=36.9, spo2=98),
        physical_exam="Multiple tense, clear-fluid-filled bullae on erythematous and normal-appearing skin. Urticarial plaques. Nikolsky sign negative. No oral mucosal lesions.",
        labs="CBC: eosinophilia (12%). BMP normal.",
        additional_workup="Skin biopsy: subepidermal blister with eosinophilic infiltrate. Direct immunofluorescence: linear IgG and C3 deposits along the basement membrane zone. Anti-BP180 and anti-BP230 antibodies positive.",
        correct_diagnosis="Bullous pemphigoid",
        differential=["Pemphigus vulgaris", "Linear IgA bullous dermatosis", "Dermatitis herpetiformis", "Drug eruption (bullous)"],
        explanation="Tense subepidermal blisters (Nikolsky-negative) with eosinophilic infiltrate, linear IgG/C3 at the BMZ on DIF, and positive anti-BP180/BP230 antibodies define bullous pemphigoid. DPP-4 inhibitors (linagliptin) are a recognized trigger.",
        dangerous_misses=["Pemphigus vulgaris (would require different treatment)", "Secondary infection of blisters"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.DERMATOLOGY,
        difficulty="hard",
        patient_age=43,
        patient_sex="F",
        chief_complaint="rapidly changing mole on the upper back",
        history_of_present_illness=(
            "The patient's partner noticed a mole on her upper back has changed significantly "
            "over 4 months -- becoming darker, larger, and developing irregular borders. She "
            "reports occasional itching and one episode of bleeding."
        ),
        past_medical_history="History of blistering sunburns in childhood. Family history of melanoma (father at age 55). Fair skin with numerous moles (>50).",
        physical_exam="1.4-cm asymmetric, multicolored (brown, black, blue-gray, red) macule/papule with irregular scalloped borders on the upper back. Satellite lesion 3 mm away. No palpable axillary lymphadenopathy.",
        additional_workup="Excisional biopsy: malignant melanoma, superficial spreading type, Breslow depth 2.3 mm, Clark level IV, ulceration present, mitotic rate 4/mm2, no lymphovascular invasion. BRAF V600E mutation detected.",
        correct_diagnosis="Malignant melanoma (superficial spreading type, stage IIB -- T3b N0 M0)",
        differential=["Dysplastic nevus", "Seborrheic keratosis", "Pigmented basal cell carcinoma", "Blue nevus"],
        explanation="ABCDE criteria met (Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolution). Excisional biopsy confirms invasive melanoma with ulceration and significant Breslow depth requiring sentinel lymph node biopsy.",
        dangerous_misses=["Nodal metastasis (requires SLNB)", "Amelanotic satellite lesions"],
    ))

    # ===== PSYCHIATRY (3) ==================================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.PSYCHIATRY,
        difficulty="moderate",
        patient_age=24,
        patient_sex="M",
        chief_complaint="auditory hallucinations, paranoid delusions, and social withdrawal for 5 months",
        history_of_present_illness=(
            "The patient's parents bring him in after 5 months of progressive behavioral change. "
            "He dropped out of college, stopped socializing, and stays in his room. He reports "
            "hearing voices commenting on his actions and believes the government is monitoring "
            "him through his phone. His speech is tangential, and he appears disheveled."
        ),
        past_medical_history="No prior psychiatric history. No substance use (confirmed by urine drug screen).",
        family_history="Mother diagnosed with schizophrenia at age 22.",
        vitals=Vitals(heart_rate=78, blood_pressure="118/74", respiratory_rate=16, temperature_c=36.7, spo2=99),
        physical_exam="Flat affect. Poor eye contact. Thought process tangential with loose associations. Auditory hallucinations present (responding to internal stimuli). Paranoid delusions regarding surveillance. No suicidal or homicidal ideation.",
        labs="CBC, BMP, TSH, B12, RPR all normal. UDS negative. MRI brain: normal.",
        correct_diagnosis="Schizophrenia (first episode, paranoid presentation)",
        differential=["Brief psychotic disorder", "Schizophreniform disorder", "Substance-induced psychotic disorder", "Bipolar I disorder with psychotic features"],
        explanation="Duration >1 month (5 months) with delusions, hallucinations, disorganized speech, and negative symptoms (flat affect, social withdrawal) meets DSM-5 criteria for schizophrenia. Duration >6 months total expected given prodromal period.",
        dangerous_misses=["Substance use disorder (always screen)", "Anti-NMDA receptor encephalitis"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.PSYCHIATRY,
        difficulty="moderate",
        patient_age=19,
        patient_sex="F",
        chief_complaint="severe fear of gaining weight, self-induced vomiting, and amenorrhea",
        history_of_present_illness=(
            "The patient's roommate called after finding her repeatedly vomiting after meals. She "
            "has lost 25 lb over 6 months (BMI now 16.2). She restricts caloric intake to 600 "
            "kcal/day and exercises 2 hours daily. She reports cold intolerance, hair loss, and "
            "has not had a period in 5 months."
        ),
        past_medical_history="Previously healthy.",
        vitals=Vitals(heart_rate=48, blood_pressure="88/54", respiratory_rate=14, temperature_c=35.6, spo2=98),
        physical_exam="Cachectic. Lanugo hair on arms and back. Parotid gland enlargement. Russell sign (calluses on dorsum of hand). Dry skin, brittle nails. Postural hypotension (sitting 88/54, standing 72/44 with lightheadedness).",
        labs="K 2.8 mEq/L, Cl 92 mEq/L, HCO3 32 mEq/L (metabolic alkalosis), BUN 28, Cr 0.9. Glucose 62 mg/dL. Amylase mildly elevated (salivary). Hgb 10.8. TSH 0.8 (low-normal). LH/FSH low. ECG: sinus bradycardia, prolonged QTc (490 ms), U waves.",
        correct_diagnosis="Anorexia nervosa (binge-purge subtype) with medical complications",
        differential=["Bulimia nervosa", "Major depressive disorder with appetite loss", "Hyperthyroidism", "Malabsorption syndrome"],
        explanation="Severe restriction with BMI <17, intense fear of weight gain, binge-purge behavior (Russell sign, parotid enlargement, hypokalemic metabolic alkalosis from vomiting), amenorrhea, and cardiac complications (bradycardia, QTc prolongation).",
        dangerous_misses=["Cardiac arrhythmia from hypokalemia/prolonged QTc", "Refeeding syndrome during treatment"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.PSYCHIATRY,
        difficulty="hard",
        patient_age=35,
        patient_sex="F",
        chief_complaint="elevated mood, decreased sleep, reckless spending, and pressured speech for 1 week",
        history_of_present_illness=(
            "The patient's husband brought her in because she has slept only 2 hours/night for the "
            "past 8 days while seeming energized, talking rapidly and jumping between topics. She "
            "has spent $15,000 on online shopping, started multiple 'business ventures,' and was "
            "found directing traffic at a busy intersection. She has a prior depressive episode."
        ),
        past_medical_history="One major depressive episode 2 years ago treated with sertraline (discontinued by patient).",
        family_history="Father with bipolar I disorder.",
        vitals=Vitals(heart_rate=96, blood_pressure="132/82", respiratory_rate=18, temperature_c=37.0, spo2=99),
        physical_exam="Appearance: brightly dressed, heavy makeup, multiple necklaces. Speech: pressured, loud, rapid. Mood: 'Fantastic, I've never felt better.' Affect: euphoric, labile. Thought process: flight of ideas. Grandiosity present ('I'm going to be a billionaire'). Distractible. No hallucinations. Denies suicidal ideation.",
        labs="UDS negative. TSH 2.1 (normal). CBC, BMP, LFTs normal. BAL 0.0.",
        correct_diagnosis="Bipolar I disorder, current episode manic (severe, without psychotic features)",
        differential=["Substance-induced mania (stimulants)", "Hyperthyroidism", "Schizoaffective disorder (bipolar type)", "ADHD with impulsive spending"],
        explanation="Distinct period of abnormally elevated/expansive mood lasting >1 week with 3+ manic symptoms (decreased need for sleep, pressured speech, flight of ideas, grandiosity, excessive spending, psychomotor agitation), causing marked functional impairment. Prior depressive episode establishes bipolar I.",
        dangerous_misses=["Psychotic features (may emerge)", "Suicidal ideation during mixed or transition states"],
    ))

    # ===== PEDIATRICS (3) ==================================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.PEDIATRICS,
        difficulty="moderate",
        patient_age=3,
        patient_sex="M",
        chief_complaint="high fever for 5 days, bilateral conjunctival injection, and rash",
        history_of_present_illness=(
            "The child has had persistent fever (39.5-40.5 C) for 5 days unresponsive to "
            "antipyretics. Parents note red eyes (no discharge), cracked red lips, a rash on his "
            "trunk, and swollen hands and feet. He is very irritable."
        ),
        past_medical_history="Full-term birth, immunizations up to date.",
        vitals=Vitals(heart_rate=148, blood_pressure="90/55", respiratory_rate=28, temperature_c=40.2, spo2=98),
        physical_exam="Bilateral non-exudative conjunctival injection. Strawberry tongue. Erythema and cracking of lips. Polymorphous maculopapular rash on trunk. Edema and erythema of hands and feet. Cervical lymphadenopathy (1.8 cm, unilateral). Irritable but consolable.",
        labs="WBC 18,400 with neutrophil predominance, Hgb 11.0, platelets 480,000. ESR 82, CRP 120 mg/L. ALT 68, albumin 2.8. Urinalysis: sterile pyuria (25 WBC/hpf). BNP 340 pg/mL.",
        imaging="Echocardiogram: mild perivascular brightness of the LAD coronary artery. No aneurysm yet. Mild mitral regurgitation.",
        correct_diagnosis="Kawasaki disease (complete, Day 5)",
        differential=["Scarlet fever", "Measles", "Systemic juvenile idiopathic arthritis", "Toxic shock syndrome"],
        explanation="Five classic criteria present: fever >5 days, bilateral conjunctival injection, oral mucosal changes (strawberry tongue, cracked lips), extremity changes (edema/erythema), rash. Cervical lymphadenopathy and coronary artery changes confirm. IVIG is urgent.",
        dangerous_misses=["Coronary artery aneurysm (requires urgent IVIG)", "Macrophage activation syndrome"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.PEDIATRICS,
        difficulty="moderate",
        patient_age=2,
        patient_sex="F",
        chief_complaint="barking cough, stridor, and respiratory distress",
        history_of_present_illness=(
            "The child developed rhinorrhea and mild cough 2 days ago, progressing to a harsh, "
            "barking cough and noisy breathing that worsened tonight. Parents describe an "
            "inspiratory 'crowing' sound. She becomes agitated when examined."
        ),
        past_medical_history="Born full-term. No prior respiratory issues. Immunizations up to date.",
        vitals=Vitals(heart_rate=140, blood_pressure="90/58", respiratory_rate=36, temperature_c=38.4, spo2=92),
        physical_exam="Inspiratory stridor at rest, barking cough. Suprasternal and intercostal retractions. No drooling. Oropharynx mildly erythematous without membrane or exudate. Bilateral air entry diminished but symmetric. No wheezing.",
        labs="Not routinely indicated. If obtained: WBC mildly elevated.",
        imaging="AP neck X-ray: subglottic narrowing -- 'steeple sign.'",
        correct_diagnosis="Moderate-to-severe croup (acute laryngotracheobronchitis, likely parainfluenza virus)",
        differential=["Epiglottitis", "Foreign body aspiration", "Bacterial tracheitis", "Retropharyngeal abscess"],
        explanation="Viral prodrome progressing to barking cough, inspiratory stridor, and steeple sign on neck X-ray in a toddler is classic for croup. Stridor at rest with retractions and hypoxia indicates moderate-to-severe disease.",
        dangerous_misses=["Epiglottitis (no drooling or toxic appearance, but always consider)", "Foreign body aspiration"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.PEDIATRICS,
        difficulty="hard",
        patient_age=8,
        patient_sex="M",
        chief_complaint="abdominal pain, bloody stools, and palpable purpura on the legs",
        history_of_present_illness=(
            "The child developed a purpuric rash on both legs and buttocks 1 week ago following "
            "an upper respiratory infection. He now reports colicky abdominal pain, one episode "
            "of bloody stool, and bilateral knee and ankle pain."
        ),
        past_medical_history="Recent URI 2 weeks ago. Otherwise healthy.",
        vitals=Vitals(heart_rate=100, blood_pressure="112/72", respiratory_rate=20, temperature_c=37.4, spo2=99),
        physical_exam="Palpable, non-blanching purpura concentrated on bilateral lower extremities and buttocks, sparing the trunk. Bilateral ankle and knee swelling with tenderness. Diffuse abdominal tenderness without peritoneal signs.",
        labs="CBC: platelets 340,000 (normal -- important to exclude ITP). BMP normal. Cr 0.5 (normal). Urinalysis: 2+ blood, 1+ protein. Coagulation studies normal. ESR 38. Serum IgA elevated. Stool guaiac positive.",
        correct_diagnosis="IgA vasculitis (Henoch-Schonlein purpura) with GI and renal involvement",
        differential=["Immune thrombocytopenic purpura (ITP)", "Meningococcemia", "Hemolytic uremic syndrome", "Polyarteritis nodosa"],
        explanation="Classic tetrad in a child: palpable purpura on lower extremities/buttocks (normal platelets ruling out ITP), arthritis, abdominal pain with GI bleeding, and hematuria/proteinuria indicating renal involvement. Elevated IgA supports diagnosis.",
        dangerous_misses=["Intussusception (most common GI complication)", "Progressive IgA nephropathy"],
    ))

    # ===== OB/GYN (3) ======================================================

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.OBSTETRICS_GYNECOLOGY,
        difficulty="moderate",
        patient_age=28,
        patient_sex="F",
        chief_complaint="sudden-onset severe right lower quadrant pain and vaginal spotting at 7 weeks gestation",
        history_of_present_illness=(
            "The patient has a positive home pregnancy test (LMP 7 weeks ago). She developed "
            "sudden sharp right lower quadrant pain 2 hours ago, rated 8/10, with light vaginal "
            "spotting. She reports mild dizziness on standing."
        ),
        past_medical_history="Prior chlamydia infection treated 3 years ago. Previous right salpingitis. IUD removed 4 months ago in anticipation of pregnancy.",
        vitals=Vitals(heart_rate=112, blood_pressure="96/62", respiratory_rate=20, temperature_c=36.8, spo2=98),
        physical_exam="RLQ tenderness with guarding. Cervical motion tenderness on bimanual exam. Right adnexal tenderness and fullness. Uterus mildly enlarged. Small amount of dark blood in the vaginal vault.",
        labs="Beta-hCG 2,400 mIU/mL. Hgb 9.8 g/dL. Type and screen: O positive.",
        imaging="Transvaginal ultrasound: empty uterus (no intrauterine gestational sac). Right adnexal complex mass (3.2 cm) with ring-of-fire sign on Doppler. Moderate free fluid in the cul-de-sac.",
        correct_diagnosis="Ruptured ectopic pregnancy (right tubal)",
        differential=["Threatened abortion", "Ovarian torsion", "Ruptured corpus luteum cyst", "Appendicitis"],
        explanation="Beta-hCG above discriminatory zone with empty uterus, adnexal mass, free fluid, and hemodynamic instability in a patient with risk factors (prior PID, salpingitis) is a ruptured ectopic pregnancy until proven otherwise.",
        dangerous_misses=["Hemorrhagic shock from ongoing rupture", "Heterotopic pregnancy"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.OBSTETRICS_GYNECOLOGY,
        difficulty="hard",
        patient_age=32,
        patient_sex="F",
        chief_complaint="severe headache, visual changes, and elevated blood pressure at 34 weeks gestation",
        history_of_present_illness=(
            "A G2P1 at 34 weeks presents with a severe, persistent headache unrelieved by "
            "acetaminophen, blurred vision with 'flashing lights,' right upper quadrant pain, "
            "and nausea. She noticed significant swelling of her face and hands over 2 days. "
            "Her blood pressure was 108/68 at her last visit 2 weeks ago."
        ),
        past_medical_history="Uncomplicated first pregnancy. No chronic hypertension.",
        vitals=Vitals(heart_rate=92, blood_pressure="172/108", respiratory_rate=18, temperature_c=36.9, spo2=97),
        physical_exam="Facial and hand edema. 3+ pitting edema bilateral lower extremities. Brisk deep tendon reflexes (3+ with sustained clonus). RUQ tenderness. Fundal height appropriate for dates. Fetal heart tones 150 bpm.",
        labs="Platelets 82,000, AST 286, ALT 312, LDH 680, Cr 1.4, uric acid 8.2. Peripheral smear: schistocytes. Urine protein/creatinine ratio 5.8 (severe proteinuria). Haptoglobin low. Fibrinogen 180.",
        correct_diagnosis="Preeclampsia with severe features and HELLP syndrome",
        differential=["Gestational hypertension", "Chronic hypertension with superimposed preeclampsia", "Thrombotic thrombocytopenic purpura (TTP)", "Acute fatty liver of pregnancy"],
        explanation="New-onset hypertension >160/110 after 20 weeks with severe features (headache, visual changes, thrombocytopenia, elevated transaminases, hemolysis with schistocytes, severe proteinuria, hyperreflexia/clonus) meets criteria for preeclampsia with severe features complicated by HELLP (Hemolysis, Elevated Liver enzymes, Low Platelets).",
        dangerous_misses=["Eclampsia (imminent seizure given hyperreflexia/clonus)", "Placental abruption", "Hepatic rupture"],
    ))

    vs.append(ClinicalVignette(
        specialty=MedicalSpecialty.OBSTETRICS_GYNECOLOGY,
        difficulty="moderate",
        patient_age=56,
        patient_sex="F",
        chief_complaint="postmenopausal bleeding and abdominal bloating",
        history_of_present_illness=(
            "The patient went through menopause at age 51 and has had no vaginal bleeding until "
            "2 months ago when she noticed intermittent spotting. She also reports progressive "
            "abdominal bloating, early satiety, and a 5-lb unintentional weight loss."
        ),
        past_medical_history="Obesity (BMI 38), hypertension, type 2 diabetes. Nulliparous. History of polycystic ovary syndrome. Tamoxifen use for 3 years after breast cancer (completed treatment).",
        vitals=Vitals(heart_rate=78, blood_pressure="142/86", respiratory_rate=16, temperature_c=36.7, spo2=98),
        physical_exam="Abdomen: mildly distended with a shifting dullness. Pelvic exam: scant blood at the cervical os, uterus bulky, left adnexal fullness.",
        labs="CA-125: 284 U/mL (elevated). CBC normal. BMP normal.",
        imaging="Transvaginal ultrasound: thickened endometrial stripe (18 mm). Left ovarian complex mass (6 cm) with solid and cystic components. CT abdomen/pelvis: omental caking, moderate ascites.",
        additional_workup="Endometrial biopsy: complex atypical hyperplasia with foci suspicious for endometrial adenocarcinoma.",
        correct_diagnosis="Endometrial carcinoma with concurrent suspicious ovarian mass (possible synchronous or metastatic disease)",
        differential=["Endometrial hyperplasia (benign)", "Ovarian cancer (primary) with endometrial involvement", "Endometrial polyp", "Cervical cancer"],
        explanation="Postmenopausal bleeding with thickened endometrial stripe, biopsy showing atypical hyperplasia/carcinoma, in a patient with multiple risk factors (obesity, nulliparity, PCOS, tamoxifen, diabetes). Concurrent ovarian mass and omental disease require surgical staging.",
        dangerous_misses=["Advanced-stage disease with peritoneal carcinomatosis", "Synchronous ovarian cancer"],
    ))

    return vs
