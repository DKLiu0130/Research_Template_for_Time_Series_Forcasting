"""Import-level smoke check for the research template."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factory import build_model
from utils.config import load_config


def main():
    cfg = load_config(ROOT / 'configs' / 'example.json')
    model = build_model(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f'Smoke OK: built {cfg["model"]["name"]} with {n} parameters')


if __name__ == '__main__':
    main()
