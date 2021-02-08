#!/usr/bin/python3
import sys
import subprocess
import argparse
import os
import shutil
import datetime

BUCKET_NAME = "mlagents"

def image_uri():
    project_id = run_shell('gcloud config list project --format value(core.project)')
    repo_name = 'mlagents'
    image_tag = 'mlagents'
    return 'gcr.io/{}/{}:{}'.format(project_id, repo_name, image_tag)


def run_shell(command, echo=True):
    if echo:
        print("+ "+command)
    process = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE)
    out = ""
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            out = out + output.decode("utf-8")
            print(output.decode("utf-8").strip("\n"))
    rc = process.poll()
    return out.strip("\n")


def create_machine(args):
    machine_name = args.name
    gpu = args.gpu
    cmd = '' \
          'gcloud compute instances create {}' \
          ' --image-family=tf-1-14-cu100' \
          ' --image-project=deeplearning-platform-release' \
          ' --maintenance-policy=TERMINATE' \
          ' --metadata=install-nvidia-driver=True' \
          ' --machine-type=n1-standard-2'.format(machine_name)
    if gpu:
        cmd += ' --accelerator=type=nvidia-tesla-v100,count=1'
    run_shell(cmd)


def attach_disk(args):
    machine_name = args.name
    cmd = '' \
          'gcloud compute instances attach-disk {}' \
          ' --disk=disk-1' \
          ' --mode=ro' \
          ' --zone=$(ZONE)'.format(machine_name)
    run_shell(cmd.replace('$(INSTANCE_NAME)', args.name).replace('$(ZONE)', args.zone))


def docker_build_push(args):
    location = args.location
    if location == 'remote':
        run_shell('gcloud builds submit --tag {} --timeout=15m'.format(image_uri()))
    elif location == 'local':
        run_shell('docker build -t {} .'.format(image_uri()))
        if args.push:
            run_shell('docker push {}'.format(image_uri()))


def ssh(args):
    machine_name = args.name
    run_shell('gcloud compute ssh {}'.format(machine_name))


def docker_run(args):
    cwd = os.getcwd()
    run_shell('docker run -it -v {}/tmp:/mnt/pwd {} /bin/bash'.format(cwd, image_uri()))


def train_remote(args):
    app_loc = args.app
    config_loc = args.config
    gcp_config = args.gcp_config
    run_shell('gsutil -m rsync -J -r {} gs://{}/train-data/app'.format(app_loc, BUCKET_NAME))
    run_shell("gsutil cp {} gs://{}/train-data/config.yaml".format(config_loc, BUCKET_NAME))
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = '{}_{}'.format('mlagents', time_str)
    cmd = '' \
          'gcloud ai-platform jobs submit training {} ' \
          '--master-image-uri=gcr.io/testeran/mlagents:mlagents ' \
          '{}' \
          '-- --job={}'.format(
                job_name,
                "--config {} ".format(gcp_config) if gcp_config else "",
                job_name)
    run_shell(cmd)
    if args.tensorboard:
        run_shell('tensorboard --logdir=gs://mlagents/results/{}'.format(job_name))


def train_local(args):
    app_loc = args.app
    config_loc = args.config
    print("training on {} with {}".format(app_loc, config_loc))
    temp_dir = 'tmp'
    shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    shutil.copytree(app_loc, '{}/app'.format(temp_dir))
    shutil.copy(config_loc, '{}/config.yaml'.format(temp_dir))
    run_shell("docker run -v {}/tmp:/mnt/pwd {} --local".format(os.getcwd(), image_uri()))


def main():
    parser = argparse.ArgumentParser(description='train mlagents locally and remotely')
    subparsers = parser.add_subparsers()

    parser_build = subparsers.add_parser('docker-build', help='build docker image')
    parser_build.add_argument('location', choices=['local', 'remote'], type=str, help='train locally or on GCP')
    parser_build.add_argument('--push', action='store_true', help='Push local builds')
    parser_build.set_defaults(func=docker_build_push)

    parser_run = subparsers.add_parser('docker-run', help='docker run')
    parser_run.set_defaults(func=docker_run)

    parser_train = subparsers.add_parser('train', help='train on image')

    subparsers_train = parser_train.add_subparsers()
    parser_train_local = subparsers_train.add_parser('local', help='train on image locally')
    parser_train_local.add_argument('--app', metavar="app-location", required=True, type=str, help='location of application')
    parser_train_local.add_argument('--config', metavar="config-file", required=True, type=str, help='location of config file')
    parser_train_local.set_defaults(func=train_local)

    parser_train_remote = subparsers_train.add_parser('remote', help='train on image remotely')
    parser_train_remote.add_argument('--app', metavar="app-location", required=True, type=str, help='location of application')
    parser_train_remote.add_argument('--config', metavar="config-file", required=True, type=str, help='location of config file')
    parser_train_remote.add_argument('--gcp-config', default="", type=str, help="hypertraining gcp config")
    parser_train_remote.add_argument('--tensorboard', action='store_true', help="start tensorboard")
    parser_train_remote.set_defaults(func=train_remote)


    parser_create_machine = subparsers.add_parser('create-machine', help='create a new vm instance')
    parser_create_machine.add_argument('name', type=str, help='the name of the machine')
    parser_create_machine.add_argument('--attach-gpu', dest="gpu", action='store_true', help='Attach a GPU')
    parser_create_machine.set_defaults(func=create_machine)

    parser_ssh = subparsers.add_parser('ssh', help='SSH to a vm instance')
    parser_ssh.add_argument('name', help='the name of the machine')
    parser_ssh.set_defaults(func=ssh)

    args = parser.parse_args()
    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments, try -h")
        # parser.print_help()

    args.func(args)


if __name__ == "__main__":
    main()
