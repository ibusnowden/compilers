"""Week 10: sharded tensors: a tuple of per-device UOps plus an axis. Elementwise ops map over shards;
reductions over the sharded axis need an allreduce."""
