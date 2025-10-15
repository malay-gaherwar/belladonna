import argparse
import json
import sys

from .config import AppConfig
from .logging_config import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Belladonna CLI")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print the resolved config and exit"
    )
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    configure_logging(cfg.log_level)

    if args.print_config:
        print(json.dumps(cfg.model_dump(), indent=2))
        sys.exit(0)

    print("✅ Belladonna CLI ok. Edit src/belladonna/__main__.py to add commands.")


if __name__ == "__main__":
    main()
