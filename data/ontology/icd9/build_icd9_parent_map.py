# Usage:
# python3 data/ontology/icd9/build_icd9_parent_map.py --cms-input data/ontology/icd9/ICD9CM_tabular.rtf --dataset-pkl data/mimiciii/mimic_hf_cohort.pkl --output data/mimiciii/icd9_parent_map.csv  --unmatched-output data/mimiciii/icd9_parent_map_unmatched.txt --require-cms 0.05 > data/mimiciii/build_icd9_log.md
import argparse
import csv
import os
import pickle
import re
import sys

ROOT_CODE = "__ROOT__"

NUMERIC_CHAPTERS = [
    ("001-139", 1, 139),
    ("140-239", 140, 239),
    ("240-279", 240, 279),
    ("280-289", 280, 289),
    ("290-319", 290, 319),
    ("320-389", 320, 389),
    ("390-459", 390, 459),
    ("460-519", 460, 519),
    ("520-579", 520, 579),
    ("580-629", 580, 629),
    ("630-679", 630, 679),
    ("680-709", 680, 709),
    ("710-739", 710, 739),
    ("740-759", 740, 759),
    ("760-779", 760, 779),
    ("780-799", 780, 799),
    ("800-999", 800, 999),
]

V_CHAPTER = "V01-V91"
E_CHAPTER = "E000-E999"

CODE_RE = re.compile(
    r"\b(?:\d{3}(?:\.\d{1,2})?|V\d{2}(?:\.\d{1,2})?|E\d{3}(?:\.\d{1,2})?)\b",
    re.IGNORECASE,
)
SUSPECT_RE = re.compile(r"\b\d{2}\b")


def normalize_code(raw):
    if raw is None:
        return ""
    code = str(raw).strip().upper()
    if not code:
        return ""
    code = "".join(code.split())
    while code.endswith("."):
        code = code[:-1]
    if not code:
        return ""

    if "." in code:
        return code

    if code.isdigit() and len(code) > 3:
        return code[:3] + "." + code[3:]

    if code.startswith("V") and len(code) > 3 and code[1:].isdigit():
        return code[:3] + "." + code[3:]

    if code.startswith("E") and len(code) > 4 and code[1:].isdigit():
        return code[:4] + "." + code[4:]

    return code


def code_match_variants(code):
    variants = {code}
    if "." not in code:
        return variants
    base, dec = code.split(".", 1)
    if not dec.isdigit():
        return variants
    variants.add(code.replace(".", ""))
    variants.add(base)

    dec = dec[:2]
    if dec:
        variants.add(f"{base}.{dec[0]}")
        if len(dec) == 2:
            variants.add(f"{base}.{dec}")
            if dec[1] == "0":
                variants.add(f"{base}.{dec[0]}")
        if len(dec) == 1:
            variants.add(f"{base}.{dec}0")
    return variants


def load_dataset_codes(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        code_map = data.get("code_map")
        if isinstance(code_map, dict):
            return list(code_map.keys())
    if hasattr(data, "code_map"):
        return list(data.code_map.keys())
    raise ValueError(f"Unsupported pickle structure: {pkl_path}")


def detect_format(path, fmt):
    if fmt != "auto":
        return fmt
    ext = os.path.splitext(path)[1].lower()
    if ext == ".rtf":
        return "rtf"
    return "txt"


def rtf_to_text(data):
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b in (123, 125):  # { }
            i += 1
            continue
        if b != 92:  # \
            out.append(b)
            i += 1
            continue

        if i + 1 >= n:
            break
        nxt = data[i + 1]
        if nxt == 39:  # \'xx
            if i + 3 < n:
                hex_bytes = data[i + 2 : i + 4]
                try:
                    out.append(int(hex_bytes, 16))
                except ValueError:
                    pass
                i += 4
                continue
            i += 2
            continue
        if nxt in (92, 123, 125):  # \\ \{ \}
            out.append(nxt)
            i += 2
            continue
        if nxt == 126:  # \~
            out.append(32)
            i += 2
            continue
        if nxt == 45:  # \-
            i += 2
            continue

        j = i + 1
        while j < n and (65 <= data[j] <= 90 or 97 <= data[j] <= 122):
            j += 1
        word = data[i + 1 : j].decode("ascii", errors="ignore").lower()

        k = j
        if k < n and data[k] in (43, 45):  # + or -
            k += 1
        while k < n and 48 <= data[k] <= 57:
            k += 1
        if k < n and data[k] == 32:
            k += 1

        if word in ("par", "line"):
            out.append(10)
        elif word == "tab":
            out.append(9)
        elif word in ("emdash", "endash"):
            out.append(45)
        i = k
    text = out.decode("latin-1", errors="ignore")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_cms_codes(path, fmt):
    fmt = detect_format(path, fmt)
    if fmt not in ("rtf", "txt"):
        raise ValueError(f"Unknown format: {fmt}")
    with open(path, "rb") as f:
        raw = f.read()
    if fmt == "rtf":
        text = rtf_to_text(raw)
    else:
        text = raw.decode("latin-1", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")

    codes = set()
    suspect_count = 0
    for line in text.splitlines():
        has_code = False
        for token in CODE_RE.findall(line):
            norm = normalize_code(token)
            if norm:
                codes.add(norm)
                has_code = True
        if has_code:
            suspect_count += len(SUSPECT_RE.findall(line))
    return codes, suspect_count


def numeric_chapter(num):
    for label, lo, hi in NUMERIC_CHAPTERS:
        if lo <= num <= hi:
            return label
    return None


def category_and_chapter(code):
    if not code:
        return None, None
    prefix = ""
    rest = code
    if code[0] in ("V", "E"):
        prefix = code[0]
        rest = code[1:]
    main = rest.split(".", 1)[0]
    if not main.isdigit():
        return None, None

    if prefix == "V":
        digits = main.zfill(2)
        return f"V{digits[:2]}", V_CHAPTER
    if prefix == "E":
        digits = main.zfill(3)
        return f"E{digits[:3]}", E_CHAPTER

    digits = main.zfill(3)
    chapter = numeric_chapter(int(digits))
    return digits[:3], chapter


def add_parent(child, parent, parent_map, nonfatal=False, conflicts=None):
    if not child or not parent or child == parent:
        return
    existing = parent_map.get(child)
    if existing is not None and existing != parent:
        msg = f"Conflicting parent for {child}: {existing} vs {parent}"
        if nonfatal:
            if conflicts is not None:
                conflicts.append(msg)
            return
        raise ValueError(msg)
    parent_map[child] = parent


def build_parent_map(dataset_codes, cms_codes, nonfatal_conflicts=False):
    parent_map = {}
    matched = set()
    unmatched = set()
    unmatched_types = {"numeric": 0, "V": 0, "E": 0, "weird": 0}
    unmatched_structural = 0
    conflicts = []

    cms_match = set()
    for code in cms_codes:
        cms_match.update(code_match_variants(code))

    cms_categories = set()
    for code in cms_codes:
        category, _ = category_and_chapter(code)
        if category:
            cms_categories.add(category)

    for code in dataset_codes:
        category, chapter = category_and_chapter(code)
        if category is None:
            unmatched.add(code)
            if code.startswith("V"):
                unmatched_types["V"] += 1
            elif code.startswith("E"):
                unmatched_types["E"] += 1
            elif code[:1].isdigit():
                unmatched_types["numeric"] += 1
            else:
                unmatched_types["weird"] += 1
            add_parent(code, ROOT_CODE, parent_map, nonfatal_conflicts, conflicts)
            continue

        if code in cms_match:
            matched.add(code)
            if code != category:
                add_parent(code, category, parent_map, nonfatal_conflicts, conflicts)
        elif category in cms_categories:
            matched.add(code)
            add_parent(code, category, parent_map, nonfatal_conflicts, conflicts)
        else:
            unmatched.add(code)
            if code.startswith("V"):
                unmatched_types["V"] += 1
            elif code.startswith("E"):
                unmatched_types["E"] += 1
            else:
                unmatched_types["numeric"] += 1
            if code != category:
                add_parent(code, category, parent_map, nonfatal_conflicts, conflicts)
            if chapter is None:
                add_parent(category, ROOT_CODE, parent_map, nonfatal_conflicts, conflicts)
            else:
                add_parent(category, chapter, parent_map, nonfatal_conflicts, conflicts)
                add_parent(chapter, ROOT_CODE, parent_map, nonfatal_conflicts, conflicts)
            unmatched_structural += 1
            continue

        if chapter is None:
            add_parent(category, ROOT_CODE, parent_map, nonfatal_conflicts, conflicts)
        else:
            add_parent(category, chapter, parent_map, nonfatal_conflicts, conflicts)
            add_parent(chapter, ROOT_CODE, parent_map, nonfatal_conflicts, conflicts)

    return parent_map, matched, unmatched, unmatched_types, unmatched_structural, conflicts


def compute_depths(parent_map):
    nodes = set(parent_map.keys()) | set(parent_map.values()) | {ROOT_CODE}
    depths = {}
    visiting = set()

    def depth(node):
        if node == ROOT_CODE:
            return 0
        if node in depths:
            return depths[node]
        if node in visiting:
            raise ValueError(f"Cycle detected at {node}")
        visiting.add(node)
        parent = parent_map.get(node, ROOT_CODE)
        d = depth(parent) + 1
        visiting.remove(node)
        depths[node] = d
        return d

    for node in nodes:
        depth(node)
    return nodes, depths


def write_csv(parent_map, output_path):
    dirpath = os.path.dirname(output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for child in sorted(parent_map.keys()):
            writer.writerow([child, parent_map[child]])


def main():
    parser = argparse.ArgumentParser(
        description="Build ICD-9 hierarchy from CMS diagnosis code titles."
    )
    parser.add_argument("--cms-input", type=str, required=True,
                        help="Path to CMS ICD-9-CM diagnosis code titles (.rtf or .txt).")
    parser.add_argument("--dataset-pkl", type=str, required=True,
                        help="Path to dataset pickle containing code_map.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path for child,parent pairs.")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "rtf", "txt"],
                        help="Input format for CMS file.")
    parser.add_argument("--require-cms", type=float, default=None,
                        help="Fail if unmatched fraction exceeds this threshold (e.g., 0.05).")
    parser.add_argument("--nonfatal-conflicts", action="store_true",
                        help="Log parent conflicts instead of raising.")
    parser.add_argument("--unmatched-output", type=str, default="",
                        help="Optional path for unmatched codes list (txt).")
    args = parser.parse_args()

    dataset_raw = load_dataset_codes(args.dataset_pkl)
    dataset_codes = []
    for code in dataset_raw:
        norm = normalize_code(code)
        if norm:
            dataset_codes.append(norm)
    dataset_codes = sorted(set(dataset_codes))
    if not dataset_codes:
        raise RuntimeError("No dataset codes found after normalization.")

    cms_codes, suspect_count = load_cms_codes(args.cms_input, args.format)
    if not cms_codes:
        raise RuntimeError("No CMS codes parsed from input file.")

    parent_map, matched, unmatched, unmatched_types, unmatched_structural, conflicts = build_parent_map(
        dataset_codes,
        cms_codes,
        nonfatal_conflicts=args.nonfatal_conflicts,
    )
    nodes, depths = compute_depths(parent_map)

    node_count = len(nodes) - 1
    depth_vals = [depths[n] for n in nodes if n != ROOT_CODE]
    max_depth = max(depth_vals) if depth_vals else 0
    mean_depth = sum(depth_vals) / len(depth_vals) if depth_vals else 0.0
    root_children = sum(1 for child, parent in parent_map.items() if parent == ROOT_CODE)
    root_fraction = root_children / node_count if node_count > 0 else 0.0

    write_csv(parent_map, args.output)

    print(f"[ICD9] Output: {args.output}")
    print(f"[ICD9] Dataset codes: {len(dataset_codes)}")
    print(f"[ICD9] Matched in CMS: {len(matched)}")
    unmatched_count = len(unmatched)
    unmatched_fraction = unmatched_count / len(dataset_codes)
    print(f"[ICD9] Unmatched: {unmatched_count}")
    print(f"[ICD9] Unmatched fraction: {unmatched_fraction:.3f}")
    print(f"[ICD9] Nodes: {node_count}")
    print(f"[ICD9] Max depth: {max_depth}")
    print(f"[ICD9] Mean depth: {mean_depth:.2f}")
    print(f"[ICD9] Root-attached fraction: {root_fraction:.3f}")
    if unmatched_count:
        print(f"[ICD9] Unmatched structurally placed: {unmatched_structural}")
    if root_fraction > 0.30:
        print("[ICD9][WARN] >30% of nodes attach directly to root.")
    if suspect_count > 50:
        print("[ICD9][WARN] FYI: many 2-digit numeric tokens found on code lines; "
              "verify this is a diagnosis titles file.")
    if args.require_cms is not None and unmatched_fraction > args.require_cms:
        raise SystemExit(
            f"[ICD9][ERROR] Unmatched fraction {unmatched_fraction:.3f} exceeds "
            f"threshold {args.require_cms:.3f}."
        )
    if unmatched_count:
        print(
            "[ICD9] Unmatched breakdown: "
            f"numeric={unmatched_types['numeric']}, "
            f"V={unmatched_types['V']}, "
            f"E={unmatched_types['E']}, "
            f"weird={unmatched_types['weird']}"
        )
    if conflicts:
        print(f"[ICD9][WARN] Parent conflicts: {len(conflicts)}")
    depth_hist = {}
    for _, depth in depths.items():
        depth_hist[depth] = depth_hist.get(depth, 0) + 1
    print(f"[ICD9] Depth histogram: {dict(sorted(depth_hist.items()))}")

    if unmatched:
        print("[ICD9][WARN] Unmatched dataset codes:", file=sys.stderr)
        for code in sorted(unmatched)[:50]:
            print(f"[ICD9][WARN] {code}", file=sys.stderr)
        if args.unmatched_output:
            unmatched_path = args.unmatched_output
        else:
            unmatched_path = args.output.replace(".csv", "_unmatched.txt")
        with open(unmatched_path, "w") as f:
            for code in sorted(unmatched):
                f.write(f"{code}\n")
        print(f"[ICD9][WARN] Full unmatched list written to {unmatched_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
