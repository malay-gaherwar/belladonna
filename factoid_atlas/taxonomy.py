#!/usr/bin/env python3
"""
BELLADONNA Factoid Atlas — clinical color-coding taxonomy.

Single source of truth for the FIVE clinically-meaningful dimensions the atlas
colors by (in addition to source / topic / year). Transcribed from the team's
"BELLADONNA Factoid Atlas" spec.

Each dimension is a controlled vocabulary; every factoid gets exactly ONE code
per dimension from the LLM classifier (classify_factoids.py). Index 0 of every
dimension is the neutral "not-specific / unclear" default, rendered grey so
off-axis factoids recede.

Exports:
    DIMENSIONS         — ordered dimension keys (== attrs.bin byte order, UI order)
    TAXONOMY           — dict: dimension -> [ {code, name, color, desc}, ... ]
    DIM_TITLES         — dimension -> human title
    label_index(dim)   — dict code -> integer index (the u8 stored in attrs.bin)
    valid_codes(dim)   — set of codes
    DRUG_NAME_SEEDS    — generic_name(lower) -> drug-class code (QA only)
    dump_json(path)    — write taxonomy.json for the web renderer
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


# Active color dimensions. Evidence/theme + Endpoint/outcome are defined below
# but intentionally left OUT of DIMENSIONS for now (re-add the keys to enable).
DIMENSIONS = ["drug_class", "biomarker", "setting"]

DIM_TITLES = {
    "drug_class": "Drug class",
    "biomarker": "Biomarker / molecular",
    "setting": "Disease setting",
    "evidence": "Evidence / theme",
    "endpoint": "Endpoint / outcome",
}

# index 0 of every dimension == grey default ("not specific / unclear").
GREY = "#3a3f4b"

TAXONOMY: Dict[str, List[dict]] = {
    # -------------------------------------------------------------------
    # 1. DRUG CLASS
    # -------------------------------------------------------------------
    "drug_class": [
        {"code": "none",           "name": "No drug / not drug-specific", "color": GREY,
         "desc": "Factoid does not center on a specific drug or drug class."},
        {"code": "endocrine",      "name": "Endocrine therapy",           "color": "#2a9d8f",
         "desc": "Aromatase inhibitors, tamoxifen, fulvestrant, oral SERDs, GnRH agonists, progestins."},
        {"code": "cdk46",          "name": "CDK4/6 inhibitor",            "color": "#e8177f",
         "desc": "Palbociclib, ribociclib, abemaciclib."},
        {"code": "pi3k",           "name": "PI3K / AKT / PTEN inhibitor",  "color": "#9b5de5",
         "desc": "Alpelisib, capivasertib, everolimus."},
        {"code": "parp",           "name": "PARP inhibitor",              "color": "#5161d6",
         "desc": "Olaparib, talazoparib."},
        {"code": "her2_mab",       "name": "Anti-HER2 antibody",          "color": "#3a86ff",
         "desc": "Trastuzumab, pertuzumab, margetuximab (naked anti-HER2 monoclonal antibodies, NOT ADCs)."},
        {"code": "adc",            "name": "Antibody-drug conjugate",     "color": "#00b4d8",
         "desc": "Trastuzumab deruxtecan, T-DM1, sacituzumab govitecan, datopotamab deruxtecan. If the drug is an ADC, classify it as ADC regardless of target."},
        {"code": "her2_tki",       "name": "HER2 / EGFR TKI",             "color": "#48cae4",
         "desc": "Tucatinib, neratinib, lapatinib."},
        {"code": "immuno",         "name": "Immunotherapy",               "color": "#52b788",
         "desc": "Checkpoint inhibitors: pembrolizumab, atezolizumab, durvalumab."},
        {"code": "chemo",          "name": "Chemotherapy",                "color": "#f4a261",
         "desc": "Taxanes, anthracyclines, platinum agents, capecitabine, gemcitabine, 5-FU, eribulin, vinorelbine, cyclophosphamide."},
        {"code": "antiangiogenic", "name": "Anti-angiogenic therapy",     "color": "#bc6c25",
         "desc": "Bevacizumab and other anti-VEGF agents."},
        {"code": "bone",           "name": "Bone-modifying agent",        "color": "#a47148",
         "desc": "Bisphosphonates (e.g. zoledronic acid), denosumab."},
        {"code": "supportive_drug","name": "Supportive drug",             "color": "#ffb703",
         "desc": "Antiemetics, G-CSF, antibiotics, analgesics, corticosteroids."},
        {"code": "other_drug",     "name": "Other drug",                  "color": "#8d99ae",
         "desc": "A specific drug/therapy that does not fit any class above."},
    ],

    # -------------------------------------------------------------------
    # 2. BIOMARKER / MOLECULAR
    # -------------------------------------------------------------------
    "biomarker": [
        {"code": "none",          "name": "No specific biomarker",       "color": GREY,
         "desc": "No biomarker mentioned / not biomarker-specific."},
        {"code": "her2_pos",      "name": "HER2-positive",               "color": "#1d4ed8",
         "desc": "HER2-positive: IHC 3+ or ISH-positive."},
        {"code": "her2_low",      "name": "HER2-low",                    "color": "#3a86ff",
         "desc": "HER2-low: IHC 1+ or IHC 2+/ISH-negative."},
        {"code": "her2_ultralow", "name": "HER2-ultralow",               "color": "#90caf9",
         "desc": "HER2-ultralow: very low HER2 expression below the HER2-low threshold."},
        {"code": "hr_pos",        "name": "HR+ / ER+ / luminal",         "color": "#2a9d8f",
         "desc": "ER-positive and/or PR-positive, luminal subtypes."},
        {"code": "tnbc",          "name": "Triple-negative (TNBC)",      "color": "#e63946",
         "desc": "ER-negative, PR-negative, HER2-negative."},
        {"code": "brca",          "name": "BRCA1/2 mutation",            "color": "#7b2cbf",
         "desc": "Germline or somatic BRCA1 or BRCA2 mutation."},
        {"code": "hered_other",   "name": "Other hereditary gene",       "color": "#9b5de5",
         "desc": "Other hereditary breast-cancer gene: PALB2, CHEK2, ATM, TP53, PTEN, CDH1."},
        {"code": "hrd",           "name": "HRD / HR deficiency",         "color": "#c77dff",
         "desc": "Homologous-recombination deficiency: genomic instability, DNA-repair deficiency (not limited to BRCA1/2)."},
        {"code": "pik3ca",        "name": "PIK3CA / AKT1 / PTEN",        "color": "#b5179e",
         "desc": "PIK3CA / AKT1 / PTEN alteration (PI3K pathway)."},
        {"code": "esr1",          "name": "ESR1 mutation",               "color": "#ff6ec7",
         "desc": "ESR1 mutation, especially in endocrine resistance or liquid biopsy."},
        {"code": "pdl1",          "name": "PD-L1 expression",            "color": "#52b788",
         "desc": "PD-L1 expression: CPS, immune-cell score, PD-L1 testing."},
        {"code": "tils",          "name": "TILs / immune infiltrate",    "color": "#80ed99",
         "desc": "Tumor-infiltrating lymphocytes, immune microenvironment."},
        {"code": "ki67",          "name": "Ki-67",                       "color": "#f4a261",
         "desc": "Ki-67 proliferation index."},
        {"code": "grade",         "name": "Tumor grade",                 "color": "#e9c46a",
         "desc": "Histologic grade, grading, differentiation."},
        {"code": "genomic_assay", "name": "Genomic recurrence assay",    "color": "#ffd166",
         "desc": "Oncotype DX, MammaPrint, EndoPredict, Prosigna/PAM50."},
        {"code": "ctdna",         "name": "ctDNA / liquid biopsy",       "color": "#00b4d8",
         "desc": "Circulating tumor DNA, plasma-based mutation testing, MRD, blood-based monitoring."},
        {"code": "other_bm",      "name": "Other biomarker",             "color": "#8d99ae",
         "desc": "Other specified biomarker not listed above."},
    ],

    # -------------------------------------------------------------------
    # 3. DISEASE SETTING
    # -------------------------------------------------------------------
    "setting": [
        {"code": "none",            "name": "Not setting-specific",      "color": GREY,
         "desc": "No specific disease stage / line / care setting."},
        {"code": "prevention",      "name": "Prevention / risk-reduce",  "color": "#52b788",
         "desc": "Chemoprevention, risk-reducing surgery, hereditary risk management."},
        {"code": "screening_dx",    "name": "Screening / diagnosis",     "color": "#2a9d8f",
         "desc": "Imaging work-up, biopsy, pathology, staging at diagnosis."},
        {"code": "neoadjuvant",     "name": "Neoadjuvant",               "color": "#f4a261",
         "desc": "Preoperative systemic therapy, pathologic complete response."},
        {"code": "adjuvant_early",  "name": "Adjuvant / early BC",       "color": "#3a86ff",
         "desc": "Post-operative therapy, curative-intent early breast cancer."},
        {"code": "surgery",         "name": "Surgery",                   "color": "#a47148",
         "desc": "Breast and axillary surgery, reconstruction."},
        {"code": "radiation",       "name": "Radiation therapy",         "color": "#d4a373",
         "desc": "Radiotherapy."},
        {"code": "locoregional_rec","name": "Locoregional recurrence",   "color": "#bc6c25",
         "desc": "Local or regional relapse and its management."},
        {"code": "metastatic",      "name": "Metastatic / advanced",     "color": "#e63946",
         "desc": "Stage IV, recurrent disease, lines of therapy."},
        {"code": "cns",             "name": "CNS / brain metastases",    "color": "#ff6ec7",
         "desc": "Brain, leptomeningeal metastases."},
        {"code": "survivorship",    "name": "Follow-up / survivorship",  "color": "#90caf9",
         "desc": "Surveillance, late effects."},
        {"code": "supportive",      "name": "Supportive care",           "color": "#ffd166",
         "desc": "Side-effect / toxicity management, supportive care."},
        {"code": "palliative",      "name": "Palliative care",           "color": "#ffb4a2",
         "desc": "Palliative and end-of-life care."},
        {"code": "ibc",             "name": "Inflammatory BC",           "color": "#c1121f",
         "desc": "Inflammatory breast cancer-specific staging, treatment, guidelines."},
        {"code": "pregnancy_young",  "name": "Pregnancy / young women",  "color": "#9b5de5",
         "desc": "Breast cancer in pregnancy, fertility preservation, young patients."},
        {"code": "elderly",         "name": "Elderly / frail",           "color": "#b5838d",
         "desc": "Geriatric oncology, frailty, dose modification."},
        {"code": "male_bc",         "name": "Male breast cancer",        "color": "#457b9d",
         "desc": "Breast cancer in men."},
    ],

    # -------------------------------------------------------------------
    # 4. EVIDENCE / THEME
    # -------------------------------------------------------------------
    "evidence": [
        {"code": "unclear",           "name": "Not evidence-specific",    "color": GREY,
         "desc": "Not evidence-specific or unclear."},
        {"code": "definition",        "name": "Definition / classification","color": "#8d99ae",
         "desc": "Definitions, disease classification, descriptive background statements."},
        {"code": "mechanism",         "name": "Mechanism / biology",      "color": "#9b5de5",
         "desc": "Mechanism of action, molecular pathways, preclinical and translational science."},
        {"code": "epidemiology",      "name": "Epidemiology / prognosis", "color": "#52b788",
         "desc": "Incidence, risk factors, prognostic statements, population-level survival statistics."},
        {"code": "rct",               "name": "RCT result",               "color": "#3a86ff",
         "desc": "Result or design of a randomized controlled trial."},
        {"code": "meta_sr",           "name": "Meta-analysis / SR",       "color": "#5161d6",
         "desc": "Meta-analysis or systematic review."},
        {"code": "observational",     "name": "Real-world / observational","color": "#ffd166",
         "desc": "Registry, cohort, case-control, real-world evidence."},
        {"code": "guideline",         "name": "Guideline recommendation", "color": "#2a9d8f",
         "desc": "Recommendation or level-of-evidence statement from AGO, ESMO, ASCO, etc."},
        {"code": "regulatory",        "name": "Regulatory / label",       "color": "#f4a261",
         "desc": "Regulatory approval, indication, or drug-label statement (FDA/EMA)."},
        {"code": "diagnostic_method", "name": "Diagnostic method",        "color": "#00b4d8",
         "desc": "Diagnostic-test performance, assay validity, imaging methodology, biomarker validation."},
        {"code": "safety_qol",        "name": "Safety / QoL",             "color": "#e63946",
         "desc": "Adverse events, toxicity, safety profile, patient-reported outcomes."},
    ],

    # -------------------------------------------------------------------
    # 5. ENDPOINT / OUTCOME
    # -------------------------------------------------------------------
    "endpoint": [
        {"code": "none",          "name": "Not outcome-specific",        "color": GREY,
         "desc": "Factoid does not report or center on a study endpoint/outcome."},
        {"code": "os",            "name": "Overall survival (OS)",       "color": "#1d4ed8",
         "desc": "Overall survival."},
        {"code": "pfs_dfs",       "name": "PFS / DFS / EFS / iDFS",      "color": "#3a86ff",
         "desc": "Progression-free, disease-free, or event-free survival (PFS, DFS, EFS, iDFS)."},
        {"code": "drfs",          "name": "Distant recurrence-free",     "color": "#48cae4",
         "desc": "Distant recurrence / metastasis-free survival (DRFS, DMFS, distant DFS)."},
        {"code": "pcr",           "name": "Pathologic CR (pCR)",         "color": "#2a9d8f",
         "desc": "Pathologic complete response (ypT0/ypN0, RCB)."},
        {"code": "orr",           "name": "Objective response (ORR)",    "color": "#52b788",
         "desc": "Objective response rate; CR, PR, overall response."},
        {"code": "toxicity",      "name": "Toxicity / adverse events",   "color": "#e63946",
         "desc": "Safety, dose reductions, treatment discontinuation as the endpoint."},
        {"code": "qol",           "name": "Quality of life / PROs",      "color": "#ff6ec7",
         "desc": "Patient-reported outcomes, symptom burden, functional status."},
        {"code": "functional",    "name": "Functional / cosmetic",       "color": "#f4a261",
         "desc": "Surgical/reconstructive outcomes, body image, physical function."},
        {"code": "locoregional",  "name": "Locoregional recurrence",     "color": "#bc6c25",
         "desc": "Local or regional relapse as the primary endpoint."},
        {"code": "dx_accuracy",   "name": "Diagnostic accuracy",         "color": "#00b4d8",
         "desc": "Sensitivity, specificity, AUC, concordance, biomarker performance."},
        {"code": "reg_approval",  "name": "Regulatory approval",         "color": "#ffd166",
         "desc": "Regulatory approval / indication as the outcome."},
    ],
}


# ---------------------------------------------------------------------------
# Derived lookups
# ---------------------------------------------------------------------------

def label_index(dim: str) -> Dict[str, int]:
    return {e["code"]: i for i, e in enumerate(TAXONOMY[dim])}


def valid_codes(dim: str) -> set:
    return {e["code"] for e in TAXONOMY[dim]}


def default_code(dim: str) -> str:
    """Index-0 code = the grey 'not specific / unclear' default."""
    return TAXONOMY[dim][0]["code"]


def color_list(dim: str) -> List[str]:
    return [e["color"] for e in TAXONOMY[dim]]


# ---------------------------------------------------------------------------
# Drug-name -> drug-class seed dictionary (QA / validation only, not used to
# classify). Built from bc_drugs_reference.csv. ADCs map to 'adc' regardless
# of target, matching the spec.
# ---------------------------------------------------------------------------

_FINE_TO_BUCKET = {
    "Aromatase inhibitor": "endocrine", "SERM": "endocrine", "SERD": "endocrine",
    "Oral SERD": "endocrine", "GnRH agonist": "endocrine", "Progestin": "endocrine",
    "CDK4/6 inhibitor": "cdk46",
    "PI3Ka inhibitor": "pi3k", "AKT inhibitor": "pi3k", "mTOR inhibitor": "pi3k",
    "PARP inhibitor": "parp",
    "Anti-HER2 mAb": "her2_mab",
    "ADC anti-HER2": "adc", "ADC anti-TROP2": "adc",
    "HER2 TKI": "her2_tki", "HER2/EGFR TKI": "her2_tki",
    "Anti-PD-1": "immuno", "Anti-PD-L1": "immuno",
    "Taxane": "chemo", "Anthracycline": "chemo", "Platinum": "chemo",
    "Antimetabolite": "chemo", "Microtubule inhibitor": "chemo", "Epothilone": "chemo",
    "Vinca alkaloid": "chemo", "Alkylating agent": "chemo",
    "Anti-VEGF": "antiangiogenic",
    "Bisphosphonate": "bone",
}


def load_drug_name_seeds(csv_path: Path) -> Dict[str, str]:
    seeds: Dict[str, str] = {}
    if not csv_path.exists():
        return seeds
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("generic_name") or "").strip().lower()
            bucket = _FINE_TO_BUCKET.get((row.get("drug_class") or "").strip())
            if name and bucket:
                seeds[name.split("(")[0].strip()] = bucket
    return seeds


_DEFAULT_CSV = Path(__file__).resolve().parents[1] / "bc_drugs_reference.csv"
DRUG_NAME_SEEDS = load_drug_name_seeds(_DEFAULT_CSV)


# ---------------------------------------------------------------------------
# Web export
# ---------------------------------------------------------------------------

def dump_json(path: Path) -> None:
    out = {
        "dimensions": [
            {
                "key": dim,
                "title": DIM_TITLES[dim],
                "labels": [
                    {"code": e["code"], "name": e["name"], "color": e["color"]}
                    for e in TAXONOMY[dim]
                ],
            }
            for dim in DIMENSIONS
        ]
    }
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for dim in DIMENSIONS:
        print(f"\n== {DIM_TITLES[dim]} ({len(TAXONOMY[dim])} labels) ==")
        for i, e in enumerate(TAXONOMY[dim]):
            print(f"  {i:2d} {e['code']:16s} {e['color']}  {e['name']}")
    print(f"\nDrug-name seeds loaded: {len(DRUG_NAME_SEEDS)}")
    out_path = Path(__file__).resolve().parent / "taxonomy.json"
    dump_json(out_path)
    print(f"Wrote {out_path}")
