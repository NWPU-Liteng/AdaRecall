from __future__ import annotations

import inspect

from src import causal_replay, graph_only_selector, query_edge_alignment


def test_primary_graph_modules_do_not_import_ann_or_baselines() -> None:
    source = "\n".join(inspect.getsource(module) for module in (causal_replay, graph_only_selector, query_edge_alignment))
    assert "import faiss" not in source
    assert "from .baselines" not in source
    assert "dense_scores" not in source
