#!/bin/bash

# Configuration
# --------------------------------------------------
CLUSTER="mahoney-${RANDOM}"
REGION="us-east1"
ZONE="us-east1-d"
DRIVER_CORES='4'
DRIVER_MEMORY='6g'
EXECUTOR_CORES='4'
EXECUTOR_MEMORY='6g'
WORKER_MEMORY='4g'


# Logging
# --------------------------------------------------
echo "==> INFO"
echo "command: $@"
echo "--> Google Cloud"
echo "cluster: $CLUSTER"
echo "region:  $REGION"
echo "zone:    $ZONE"
echo "--> Spark"
echo "spark.driver.cores:         $DRIVER_CORES"
echo "spark.driver.memory:        $DRIVER_MEMORY"
echo "spark.executor.cores:       $EXECUTOR_CORES"
echo "spark.executor.memory:      $EXECUTOR_MEMORY"
echo "spark.python.worker.memory: $WORKER_MEMORY"
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
PROPERTIES="spark.driver.cores=$DRIVER_CORES"
PROPERTIES+=",spark.driver.memory=$DRIVER_MEMORY"
PROPERTIES+=",spark.executor.cores=$EXECUTOR_CORES"
PROPERTIES+=",spark.executor.memory=$EXECUTOR_MEMORY"
PROPERTIES+=",spark.python.worker.memory=$WORKER_MEMORY"
gcloud dataproc jobs submit pyspark \
	--cluster $CLUSTER \
	--region $REGION \
	--driver-log-levels root=FATAL \
	--properties $PROPERTIES \
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