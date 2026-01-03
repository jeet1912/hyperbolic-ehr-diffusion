# ICD-9 Diagnostic Ontology (Dataset-Induced)

This folder contains the resources and scripts used to construct a **dataset-induced ICD-9 diagnostic ontology** from the official ICD-9-CM tabular specification. The resulting hierarchy is used to evaluate **hierarchical faithfulness** of learned latent representations (e.g., Tree–Latent correlation, distortion vs. depth).

---

## Conceptual Overview

ICD-9-CM defines a canonical hierarchical organization of diagnosis codes:
- **Chapters** (e.g., 001–139: Infectious and Parasitic Diseases)
- **Blocks / Sections** (numeric ranges within chapters)
- **3-digit categories** (e.g., `026`)
- **Decimal subcategories** (e.g., `026.1`)

Rather than inventing a custom hierarchy, we **project the official ICD-9-CM ontology onto the dataset** by:
1. Parsing the full ICD-9-CM *Diagnosis Tabular List*
2. Retaining only diagnosis codes observed in the dataset
3. Preserving all ancestor nodes up to a virtual root

This yields a **dataset-specific induced ontology** that is:
- Clinically grounded
- Acyclic and interpretable
- Suitable for quantitative hierarchy-faithfulness analysis

---

## Folder Structure

icd9/
├── source/
│ ├── ICD9CM_tabular.rtf # Official ICD-9-CM Diagnosis Tabular List (raw)
│ └── ICD9CM_tabular.txt # Plain-text conversion of the tabular list
│
├── induced/
│ ├── icd9_dataset_tree.csv # Final child,parent edges used by models
│ ├── icd9_nodes.csv # Optional: node metadata (type, depth)
│ └── icd9_stats.json # Sanity statistics (depth, coverage, etc.)
│
├── scripts/
│ ├── convert_rtf_to_txt.py # RTF → text conversion
│ ├── parse_icd9_tabular.py # Parse tabular list into a full ontology
│ └── induce_dataset_tree.py # Restrict ontology to dataset codes
│
└── README.md

---

## Data Sources

- **ICD-9-CM Diagnosis Tabular List**
  - Source: CDC / NLM (2011 release, ICD-9-CM v32)
  - Format: Rich Text Format (`.rtf`)
  - Contains official chapter, block, category, and subcategory structure

> **Note:** CMS “code title” files are *not* sufficient for hierarchy construction and are not used for this process.

---

## Ontology Construction Pipeline

### 1. RTF → Text Conversion
The official tabular list is provided as `.rtf`. This is converted to plain text prior to parsing:
- Preferred (macOS): `textutil -convert txt`
- Fallback: Python RTF stripping utilities

### 2. Tabular Parsing
The plain-text tabular list is parsed using structural cues:
- Chapter headers (numeric ranges, e.g., `(001–139)`)
- Block / section headers (range lines within chapters)
- 3-digit ICD categories
- Decimal ICD subcategories

Non-hierarchical annotations such as `Includes:` and `Excludes:` are ignored.

### 3. Dataset Induction
Given a dataset containing observed ICD-9 diagnosis codes:
- Only dataset-observed codes are retained
- All ancestors up to a virtual root `__ROOT__` are preserved
- The result is a connected, acyclic ontology tailored to the dataset

---

## Output Format

### `icd9_dataset_tree.csv`

A two-column CSV representing parent–child relationships:

child,parent
026.1,026
026,OTHER_BACTERIAL_DISEASES_030_041
OTHER_BACTERIAL_DISEASES_030_041,INFECTIOUS_AND_PARASITIC_DISEASES_001_139
INFECTIOUS_AND_PARASITIC_DISEASES_001_139,ROOT

This format is consumed directly by hierarchy-faithfulness metrics.

---

## Validation & Sanity Checks

The induction process reports:
- Total node count
- Maximum and mean depth
- Fraction of nodes attached directly to root
- Coverage of dataset diagnosis codes

Expected properties:
- Maximum depth ≈ 4–6
- Low root-attachment fraction
- No cycles

---

## Usage in Experiments

The induced ontology is used to compute:
- **Tree–Latent Spearman ρ**
- **Diffusion–Latent Spearman ρ**
- **Distortion vs. Depth**

These metrics quantify how well learned representations preserve known clinical hierarchy.

---

## Terminology

We refer to this process as:

> **Dataset-Induced ICD-9 Diagnostic Ontology Construction**

This emphasizes that the hierarchy is:
- Derived from a canonical standard
- Restricted to dataset-relevant structure
- Not heuristically reconstructed

---

## Notes

- This folder is designed to be extensible (e.g., ICD-10, SNOMED-CT).
- The ontology is versioned implicitly by the ICD-9-CM release year (2011).

---

## License & Attribution

ICD-9-CM is maintained by the U.S. Centers for Disease Control and Prevention (CDC) and the National Library of Medicine (NLM). This repository contains only derived structural representations for research purposes.


