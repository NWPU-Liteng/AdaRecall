# Physical Tile and End-to-end Gates

Run: `v30_layout_io_e2e_20260719`

The run rebuilt and hash-verified all 148 v28 causal snapshots, evaluated all
1,355 queries, and preserved every Semantic Graph-only candidate list. Physical
byte components included alignment padding and summed exactly to the bytes
read. Candidate identity and byte-conservation violations were zero.

Semantic layout passed Gate E decisively: mean activated Tiles fell from 16.81
to 4.48 (-73.36%), P95 from 24 to 8 (-66.67%), mean fetched bytes from 3.94 MB
to 1.04 MB (-73.53%), and requested-cold mean read time from 8.35 to 2.48 ms
(-70.30%). Dual also passed: Tiles -63.96%, bytes -64.94%, and cold mean
latency -11.22%.

Co-retrieval layout failed Gate E. It reduced mean Tiles and bytes by only about
4% and increased requested-cold mean latency by 16.02%. This is a retained
negative Access result. Dual is also weaker than Semantic on all principal
Tile metrics, so the system contribution is primarily semantic layout.

Both Graph-only methods passed Gate F. Semantic averaged 1,901.16 ms end to end
and Dual 1,891.69 ms, versus Dense 1,867.67 ms, IVF 1,884.51 ms, and Full-128
7,342.28 ms. Graph is therefore slower than Dense/IVF by 7-34 ms, but its Gate
D quality is higher on every reported metric; neither baseline strictly
dominates it. Relative to Full-128, Semantic and Dual save about 5.44-5.45
seconds on average. This is a quality-system Pareto result, not a claim that
Graph is the lowest-latency 32-candidate retriever.

Cold reads use `POSIX_FADV_DONTNEED` requests and are reported as requested-cold
OS measurements, not guaranteed raw-device cache eviction. Gate G incremental
replay remains pending, and fresh remains locked.
