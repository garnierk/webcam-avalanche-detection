from pathlib import Path

ROOT = Path(".data/annotations")
NC = 4  # classes autorisées: 0,1,2,3

def check_label_file(txt_path, nc=NC):
    errors = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append((txt_path, lineno, "bad_format", line))
                continue

            try:
                cls = int(parts[0])
            except Exception:
                errors.append((txt_path, lineno, "bad_class_parse", line))
                continue

            if not (0 <= cls < nc):
                errors.append((txt_path, lineno, "class_out_of_range", line))

            try:
                x, y, w, h = map(float, parts[1:])
            except Exception:
                errors.append((txt_path, lineno, "bad_bbox_parse", line))
                continue

            if any(v < 0 or v > 1 for v in (x, y, w, h)):
                errors.append((txt_path, lineno, "bbox_not_normalized", line))

            if w <= 0 or h <= 0:
                errors.append((txt_path, lineno, "non_positive_wh", line))

            if x - w/2 < 0 or x + w/2 > 1 or y - h/2 < 0 or y + h/2 > 1:
                errors.append((txt_path, lineno, "bbox_outside_image", line))

    return errors

all_errors = []
txt_files = sorted(ROOT.rglob("*.txt"))

print(f"{len(txt_files)} fichiers .txt trouvés sous {ROOT}")

for txt in txt_files:
    all_errors.extend(check_label_file(txt))

if not all_errors:
    print("OK: aucun problème détecté")
else:
    print(f"{len(all_errors)} erreur(s) détectée(s)\n")
    for path, lineno, kind, line in all_errors[:200]:
        print(f"[{kind}] {path}:{lineno} -> {line}")

    if len(all_errors) > 200:
        print(f"\n... {len(all_errors)-200} autres erreurs non affichées")
