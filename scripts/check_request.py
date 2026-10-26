"""Check a CreateTrainingJob or CreateProcessingJob JSON request against the account's rules.

    python check_request.py train-xgboost.json

Reads the same guardrails.json the account enforces (next to this file) and prints what
would be refused and why, before `aws sagemaker create-training-job --cli-input-json` sends it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers  # noqa: E402


def main(path):
    with open(path) as f:
        req = json.load(f)
    tags = req.get("Tags", [])
    res = req.get("ResourceConfig") or (req.get("ProcessingResources") or {}).get("ClusterConfig") or {}
    cap = (req.get("StoppingCondition") or {}).get("MaxRuntimeInSeconds")
    if "ProcessingJobName" in req and cap is None:
        print("this launch would be refused by the account:\n  - Processing job without a runtime cap: add StoppingCondition.MaxRuntimeInSeconds")
        return 1
    try:
        helpers.check_launch(tags, res.get("InstanceType"), cap)
    except ValueError as e:
        print(e)
        return 1
    print(f"{os.path.basename(path)}: tags, instance type and cap pass the account's rules")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
