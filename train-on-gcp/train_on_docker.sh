echo "sh input parameters"
echo $@
#nvidia-smi
#gsutil -m cp -r gs://mlagents/train-data/app /mnt/pwd/.
#gsutil -m cp gs://mlagents/train-data/config.yaml /mnt/pwd/.
#chmod u+x /mnt/pwd/app/rollerball_linux.x86_64
#xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' \
#  mlagents-learn /mnt/pwd/config.yaml \
#  --env=/mnt/pwd/app/rollerball_linux.x86_64 \
#  --run-id=run
#gsutil cp -r results/run gs://mlagents/results/$1
xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' python3 train_on_docker.py $@
