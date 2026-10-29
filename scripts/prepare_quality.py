from pathlib import Path
import csv
import xml.etree.ElementTree as ET
import zipfile

root = Path(__file__).resolve().parents[1]
ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
checks = {}
for name in ["grading.xlsx", "grading_retry.xlsx"]:
    with zipfile.ZipFile(root / "data/raw" / name) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(x.itertext()) for x in tree.findall("m:si", ns)]
        tree = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in tree.findall("m:sheetData/m:row", ns):
        values = {}
        for cell in row:
            column = "".join(x for x in cell.get("r") if x.isalpha())
            value = cell.find("m:v", ns)
            inline = cell.find("m:is", ns)
            value = value.text if value is not None else ""
            if inline is not None:
                value = "".join(inline.itertext())
            if cell.get("t") == "s":
                value = strings[int(value)]
            values[column] = value
        rows.append(values)
    header = rows[0]
    groups = {}
    for row in rows[1:]:
        record = {header[k]: v for k, v in row.items() if k in header}
        pid, value = record.get("prolific_pid", ""), record.get("qcheck", "")
        if len(pid) != 64 or not value:
            continue
        value = float(value)
        groups.setdefault(pid, []).append(1 if value == 3 else value)
    for pid, values in groups.items():
        checks.setdefault(pid, (min(values), max(values)))
with (root / "data/derived/quality_checks.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["prolific_pid", "minqcheck", "maxqcheck"])
    writer.writerows((pid, *values) for pid, values in sorted(checks.items()))
