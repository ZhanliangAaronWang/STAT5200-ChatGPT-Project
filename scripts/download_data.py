from pathlib import Path
import hashlib
import json
import urllib.request

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "data/source_manifest.json").read_text())
for item in manifest:
    target = root / item["file"]
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]:
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(item["url"], temporary)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
        temporary.unlink()
        raise ValueError(f"SHA-256 mismatch: {item['file']}")
    temporary.replace(target)
    print(item["file"])
