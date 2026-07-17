# A4 Controller Overhead Positioning

O1 formal rows: 480
Mean controller intent-generation latency: 7037.96 ms
Mean retrieval-selection latency after intent generation: 147.57 ms
Controller/retrieval latency ratio: 47.69x

The Qwen3-4B controller is used only to reconstruct and instrument an offline stage-aware workload for motivation analysis. It is not an additional online component of the final runtime design.

The controller latency is reported explicitly and is not hidden or subtracted from the trace.
