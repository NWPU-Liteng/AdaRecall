# MemoryCD Data Provenance

日期：2026-06-22

The user provided the MemoryCD release files in the workspace root as `meta/` and `users/`. They were moved into:

```text
agent-memory-profiling/data/public_datasets/memorycd/
```

No synthetic data, LoCoMo, MSC, LongMemEval, or self-built Amazon review data were used for V5.1.

## File Hashes

```text
8edebfecbd7e05e6461c6dfd86296419eca7e1119c1b534985097b7d0bad390a  users/cross_domain_users_sampled.jsonl.gz
8747682e3a0ad457894bce72efbb7621b8c4936f4cc6f27ec53fa508f11d9ca7  meta/Beauty_and_Personal_Care.jsonl.gz
1f50242e9d46ff84410830c2413c2721da2f20c029c028e31489bbf823a19b00  meta/Books.jsonl.gz
a843823de7280ff3d9d8b8af427a60687af558ec6d250a6069dcb55355513b2b  meta/Electronics.jsonl.gz
649771c61f8b089d18ee203225b8c8fba68e0c30554a1629fa7842c76b36b268  meta/Home_and_Kitchen.jsonl.gz
```

## Official Code

The official MemoryCD repository was previously cloned to:

```text
third_party/MemoryCD
```

Commit:

```text
4cee5ebbf8a1ac4c933bde147006217e04678370
```

