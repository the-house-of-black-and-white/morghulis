#!/usr/bin/env bash

PROJECT=$1
INSTANCE_NAME=$2
ZONE=$3

gcloud beta compute --project ${PROJECT} instances create ${INSTANCE_NAME} --zone ${ZONE} --machine-type "n1-standard-4" --subnet "default" --maintenance-policy "TERMINATE" --scopes "https://www.googleapis.com/auth/cloud-platform" --accelerator type=nvidia-tesla-k80,count=1 --min-cpu-platform "Automatic" --tags "http-server" --image "ubuntu-1604-xenial-v20171212" --image-project "ubuntu-os-cloud" --boot-disk-size "50" --boot-disk-type "pd-ssd" --boot-disk-device-name "${INSTANCE_NAME}-disk" --metadata-from-file startup-script=startup.sh
