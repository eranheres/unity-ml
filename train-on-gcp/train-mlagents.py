#!/usr/bin/python3
import sys
import subprocess
import argparse

def run_shell(command, echo=True):
    if echo:
        print(command)
    result = subprocess.run(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Print the stdout and stderr
    if len(result.stdout)>0:
        print(result.stdout.decode("utf-8").strip("\n"))
    if len(result.stderr)>0:
        print(result.stderr.decode("utf-8"))
    return result.stdout.decode("utf-8").strip("\n")

parser = argparse.ArgumentParser(description='train mlagents locally and remotely')
parser.add_argument('command', choices=['build', 'push-docker'], nargs="+")
parser.add_argument('--local', action="store_false")
args = parser.parse_args()
command = args.command[0]

project_id = run_shell('gcloud config list project --format value(core.project)')
repo_name = 'mlagents'
image_tag = 'mlagents'
image_uri = 'gcr.io/{}/{}:{}'.format(project_id, repo_name, image_tag)

if command == 'build':
    run_shell('docker build -t {} .'.format(image_uri))

if command == 'push-docker':
    run_shell('docker push {}'.format(image_uri))