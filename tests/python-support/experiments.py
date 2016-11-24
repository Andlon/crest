import os
import subprocess
import json


def _experiment_runner_path():
    this_path = os.path.dirname(os.path.realpath(__file__))
    return this_path + "/../../target/release/experiments"


def run_experiment(params):
    args = [_experiment_runner_path(), params['T'], params['h'], params['N']]
    result = subprocess.run(args=args,
                            input=None,
                            stdout=subprocess.PIPE,
                            check=True,
                            universal_newlines=True)
    return json.loads(result.stdout)


def run_experiments(param_collection):
    return [run_experiment(param) for param in param_collection]
