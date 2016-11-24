import os
import subprocess
import json


def _experiment_runner_path():
    this_path = os.path.dirname(os.path.realpath(__file__))
    return this_path + "/../../target/release/experiments"


def run_experiment(params):
    args = [_experiment_runner_path() ]
    result = subprocess.run(args=args,
                            input=json.dumps(params, indent=4),
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
    return json.loads(result.stdout)


def run_experiments(param_collection):
    return [run_experiment(param) for param in param_collection]
