# Week 2 assignment — loads, stores & memory

**Done when:** `pytest tests/week{01..02}` passes.

## Deliverables

1. `BUFFER`, `STORE`, `LOAD` UOps, typed and parsed (`buffer DT N`, `buffer DT [list]`, `store`, `load`).
2. Compile-time bounds checking for every load and store, using week-1 intervals after simplification.
3. `compiler/uc/linearize.py`: toposort with write-after-read edges. Detect when no in-place order exists and copy.
4. `compiler/uc/interp.py`: run the linear program, with all versions of a buffer aliasing one storage.
   `run_uir(src, backend="interp")` is the default this week.
5. **Your tests:** at least **12** new `.uir` files in `tests/week02/mine/`, passing the oracle. Include:
   a fork, a read of an old version, a store-to-load-forwarding case where forwarding must **not** fire (the indices
   may be equal), and 3 bounds errors.

## Checklist

- [ ] `pytest tests/week{01..02}` passes
- [ ] Shuffling the order of independent statements in any week-2 test doesn't change the result
- [ ] `DEBUG=1` prints the linearized program

## Stretch

* Store-to-load forwarding and dead-store elimination as patterns, guarded by proofs.
* A random-program fuzzer comparing your interpreter with `reference/oracle.py`.
