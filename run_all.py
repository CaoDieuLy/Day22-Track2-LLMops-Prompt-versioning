import argparse
import subprocess
import sys


STEPS = {
    "1": "01_langsmith_rag_pipeline.py",
    "2": "02_prompt_hub_ab_routing.py",
    "3": "03_ragas_evaluation.py",
    "4": "04_guardrails_validator.py",
}


def run_step(step: str) -> int:
    script = STEPS[step]
    print(f"\n=== Running step {step}: {script} ===")
    return subprocess.call([sys.executable, script])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=STEPS.keys(), help="Run one lab step")
    args = parser.parse_args()

    selected = [args.step] if args.step else list(STEPS.keys())
    for step in selected:
        code = run_step(step)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
