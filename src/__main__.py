"""CLI for nidana."""
import sys, json, argparse
from .core import Nidana

def main():
    parser = argparse.ArgumentParser(description="Nidana — LLM Clinical Reasoning Benchmark. Testing AI diagnostic accuracy across 20 medical specialties with 2000+ clinical vignettes.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Nidana()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.process(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"nidana v0.1.0 — Nidana — LLM Clinical Reasoning Benchmark. Testing AI diagnostic accuracy across 20 medical specialties with 2000+ clinical vignettes.")

if __name__ == "__main__":
    main()
