# Blocked: Official MedMemoryBench Acquisition

The v3 instruction requires official acquisition from:

- source repository: `https://github.com/AQ-MedAI/MedMemoryBench.git`
- source dataset: `Cyan27/MedMemoryBench`

The current Codex execution environment has network disabled (`CODEX_SANDBOX_NETWORK_DISABLED=1` was observed earlier in this session), and no official MedMemoryBench data files are present under `agent-memory-profiling/data/external/MedMemoryBench`.

No unofficial mirror, reconstructed dataset, synthetic substitute, cached answers, or manually fabricated schema was used.

## Required server-side commands

```bash
conda activate agent-memory-profiling
mkdir -p external
cd external
git clone https://github.com/AQ-MedAI/MedMemoryBench.git
cd MedMemoryBench
git rev-parse HEAD > ../medmemorybench_git_revision.txt
cd ../..
huggingface-cli download Cyan27/MedMemoryBench \
  --repo-type dataset \
  --local-dir agent-memory-profiling/data/external/MedMemoryBench \
  --local-dir-use-symlinks False
```

After the files are present, run:

```bash
cd agent-memory-profiling/versions/20260704_method1_motivation_medmemorybench_v3
python scripts/probe_medmemorybench_schema.py \
  --data-root ../../../data/external/MedMemoryBench \
  --language en \
  --output-root .
```
