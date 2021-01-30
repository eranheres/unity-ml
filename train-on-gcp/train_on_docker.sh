echo "train_on_docker input parameters:"
echo $@
gsutil cp -r gs://mlagents/app /mnt/pwd/.
gsutil cp gs://mlagents/config.yaml /mnt/pwd/.
xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' \
  mlagents-learn /mnt/pwd/config.yaml \
  --env=/mnt/pwd/app/rollerball_linux.x86_64 \
  --run-id=run1