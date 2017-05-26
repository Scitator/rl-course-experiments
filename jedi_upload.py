#!/usr/bin/env python

"""
Use your power, Luke!
"""

import gym
import argparse
import json
from glob import glob


def force_upload(monitor_dir, correct_name, api_key):
    f_name = glob("{}/*manifest.json".format(monitor_dir))[0]
    with open(f_name, "r") as fin:
        data = json.load(fin)
    data["env_info"]["env_id"] = correct_name
    with open(f_name, "w") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)
    gym.upload(monitor_dir, api_key=api_key)


def _parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--monitor_dir',
                        type=str)
    parser.add_argument('--correct_name',
                        type=str)
    parser.add_argument('--api_key',
                        type=str)

    args, _ = parser.parse_known_args()
    return args


def main():
    args = _parse_args()
    force_upload(args.monitor_dir, args.correct_name, args.api_key)

if __name__ == '__main__':
    main()
