"""Read the profiles attached to final display results; no metric computation."""
from pathlib import Path
import hashlib
import json
RESULTS = Path(__file__).resolve().parents[1] / "figures/data/unified_results.json"
def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def profile(name):
    registry = json.loads(RESULTS.read_text())["protocol_registry"]
    matches = [config for config in registry.values() if
               ("strict600" in config["protocol_version"] if name == "strict600" else "overflow601" in config["protocol_version"])]
    if len(matches) != 1 or fingerprint(matches[0]) not in registry:
        raise ValueError("Missing or inconsistent stored result profile: " + name)
    return matches[0]
