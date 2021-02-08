import argparse
import sys
import subprocess
import hypertune
import re
import selectors
import io
import yaml
from functools import reduce
import operator
import os
import json
import asyncio


def run_shell(command, echo=True, output_callback=None):
    if echo:
        print("> +" + command)
    p = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = ""
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ)
    sel.register(p.stderr, selectors.EVENT_READ)

    while True:
        for key, _ in sel.select():
            data = key.fileobj.read1().decode()
            out = out + data
            if not data:
                return out.strip("\n")
            if output_callback:
                output_callback(data)
            if key.fileobj is p.stdout:
                print(">", data, end="")
            else:
                print(">>", data, end="")


def download_train_data():
    run_shell('gsutil -m cp -r gs://mlagents/train-data/app /mnt/pwd/.')
    run_shell('gsutil -m cp gs://mlagents/train-data/config.yaml /mnt/pwd/.')
    run_shell('chmod u+x /mnt/pwd/app/rollerball_linux.x86_64')


async def run_mlagents(job, trial):
    def logout_callback(line):
        m = re.match(r".*Step: (\d+).*Mean Reward: (\d+\.\d+).*", line)
        if m:
            steps = m.group(1)
            reward = m.group(2)
            print("feeding hypertune with steps:{} reward:{}".format(steps, reward))
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='reward',
                metric_value=reward,
                global_step=steps)
            asyncio.create_task(async_upload_results(job=job, trial=trial))

    print("training on docker call")
    cmd = "mlagents-learn /mnt/pwd/config.yaml " \
          "--env=/mnt/pwd/app/rollerball_linux.x86_64 " \
          "--run-id=run"
    run_shell(cmd, output_callback=logout_callback)


def upload_results(job, trial):
    run_shell('gsutil rsync -J -r results/run gs://mlagents/results/{}/{}'.format(job, trial))


async def async_upload_results(job, trial):
    upload_results(job, trial)


def restamp_hypertune(yaml_file, args, dest_yaml=None):
    def getFromDict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)

    def setInDict(dataDict, mapList, value):
        getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

    def cast_to_type(val):
        try:
            return int(val)
        except:
            pass
        try:
            return float(val)
        except:
            return val

    if not dest_yaml:
        dest_yaml = yaml_file
    params = [x.split('=') for x in args]
    with open(yaml_file, 'r') as stream:
        data = yaml.safe_load(stream)
    for param in params:
        map_list = param[0][2:].split("-")
        setInDict(data, map_list, cast_to_type(param[1]))
    with io.open(dest_yaml, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)


def get_trial_num():
    tf_config_str = os.environ.get('TF_CONFIG')
    tf_config_dict = json.loads(tf_config_str)

    # Convert back to string just for pretty printing
    print("TF_CONFIG:")
    print(json.dumps(tf_config_dict, indent=2))
    return tf_config_dict["task"]["trial"]


def main():
    parser = argparse.ArgumentParser(description='train mlagents locally and remotely')
    parser.add_argument('--job', type=str, help='job name')
    parser.add_argument('--local', action='store_true', help='job name')
    args, hypertune_params = parser.parse_known_args()

    trial_num = 0
    if not args.local:
        download_train_data()
    if hypertune_params:
        print("got hypertune parameters:")
        print(hypertune_params)
        restamp_hypertune(yaml_file="/mnt/pwd/config.yaml", args=hypertune_params)
        trial_num = get_trial_num()
    asyncio.run(run_mlagents(job=args.job, trial=trial_num))
    if not args.local:
        upload_results(job=args.job, trial=trial_num)


if __name__ == "__main__":
    print("got parameters:")
    print(sys.argv)
    main()

