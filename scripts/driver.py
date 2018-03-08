#!/usr/bin/env python3
import mahoney.__main__
from knit.dask_yarn import DaskYARNCluster
from dask.distributed import Client

cluster = DaskYARNCluster()
cluster.start(nworkers=10, memory=4096, cpus=2)
client = Client(cluster)

mahoney.__main__.main()
