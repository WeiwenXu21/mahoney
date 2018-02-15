#!/bin/bash

# Configuration
# --------------------------------------------------
CLUSTER="mahoney-${RANDOM}"
REGION="us-east1"
ZONE="us-east1-d"


# Logging
# --------------------------------------------------
echo "==> INFO"
echo "command: $@"
echo "--> Google Cloud"
echo "cluster: $CLUSTER"
echo "region:  $REGION"
echo "zone:    $ZONE"
echo


# Compile
# --------------------------------------------------
echo "==> Compiling the module"
./setup.py bdist_egg
echo


# Provision
# --------------------------------------------------
echo "==> Provisioning cluster $CLUSTER in $ZONE ($REGION)"
gcloud dataproc clusters create $CLUSTER \
	--region $REGION \
	--zone $ZONE \
	--worker-machine-type n1-standard-4 \
	--num-workers 4 \
	--initialization-actions gs://cbarrick/dataproc-bootstrap.sh
echo


# Submit
# --------------------------------------------------
echo "==> Submitting job: $@"
gcloud dataproc jobs submit pyspark \
	--cluster $CLUSTER \
	--region $REGION \
	--driver-log-levels root=FATAL \
	--py-files ./dist/mahoney-*.egg \
	./scripts/driver.py \
	-- $@
echo


# Teardown
# --------------------------------------------------
echo "==> Tearing down the cluster"
yes | gcloud dataproc clusters delete $CLUSTER \
	--region $REGION \
	--async
echo
