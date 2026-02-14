# GNN product loop audit

**Core claim to test:** The graph is the system’s substrate; the GNN produces embeddings + risk that drive detection, investigation, watchlists, similar incidents, and continual improvement.

**Finding:** The GNN exists in the repo and is *used when available*, but production outputs are still partially rule-based or placeholder. If you remove the GNN, the demo outcome is largely unchanged. So today it’s **“a GNN in the repo,” not yet “a GNN-powered system.”**

---

## Litmus test: delete the GNN

If the checkpoint is missing or GNN inference is disabled:

| Surface | Without GNN | Why |
|--------|-------------|-----|
| **Detection** | Same | Agent uses `combined = rule_score` when `model_scores` is empty; 100% motif-based. |
| **Risk scores** | Same (different values) | Pipeline uses placeholder `0.1 + (i % 3) * 0.2`; agent uses motif-only scores. |
| **Watchlists** | Same | Built from scores + entity hashes + motif keywords; no embedding centroids. |
| **Similar incidents** | “Works” but meaningless | Fallback embedding = `[score, severity, node_index] + zeros`; cosine similarity is not semantically meaningful. |
| **Explanations** | Same | `model_subgraph` is a stub (single node + empty edges), not GNN-derived; motifs still drive summary. |
| **Continual improvement** | Same (none) | `finetune_last_layer_stub` does not train; feedback is stored but not used to update the model. |

So: **you can delete the GNN and get basically the same demo** — same flows, same UI, same persistence; only the *meaning* of scores and similar incidents degrades. That’s the signal the GNN is not yet the core engine.

---

## Per-surface status

### 1. Detection

- **Pipeline** (`pipeline.py`): Tries HGT inference; on failure → placeholder scores (no embeddings).
- **Agent** (`financial_security_agent._detect_risk_patterns`): `combined = 0.6 * rule_score + 0.4 * model_score` when model runs; **else `combined = rule_score`**. So detection is GNN-aware but rule-sufficient.

### 2. Investigation (explanations)

- **Pipeline** (`generate_explanations`): Uses motif tags + timeline; `model_subgraph` is **not** from the GNN — it’s `{ nodes: [node_index, type, score], edges: [] }`. No real GNN explainer wired in.

### 3. Watchlists

- **Pipeline** (`synthesize_watchlists`): Doc says “embedding centroids”; implementation uses **score + node_index** only. No centroid of GNN embeddings.
- **Agent** (`_watchlist_synthesis`): Entity patterns (canonical_hash, entity_type) + keyword patterns from motif tags. No embeddings.

### 4. Similar incidents

- **Data flow:** Embeddings come from pipeline/worker. When GNN runs, they’re the model’s pooled representation; when it doesn’t, worker uses `[score, severity, node_index] + zeros`.
- **Similarity:** `explain_service.get_similar_incidents` does cosine over stored embeddings — so it’s *embedding-based* but embeddings are only meaningful when the GNN ran.

### 5. Continual improvement

- **Feedback:** Stored in `feedback`; used for similar-incident *outcomes* (confirmed_scam / false_positive).
- **Model update:** `ml/continual/finetune_last_layer.py` is a **stub** — loads feedback, does not finetune or save a new checkpoint. No loop from feedback → model.

---

## What would make it a GNN-driven product loop

1. **Detection**  
   - Require model scores for primary signal type (e.g. relational_anomaly); if no checkpoint, either fail closed (no score) or clearly label as “rule-only, no model.”

2. **Explanations**  
   - Wire a real GNN explainer (e.g. subgraph / attention) into `generate_explanations` so `model_subgraph` and summary reflect the model, not a stub.

3. **Watchlists**  
   - Add at least one watchlist type that uses **embedding centroids** (e.g. cluster high-risk entity embeddings; store centroid or rep and match new entities by embedding distance).

4. **Similar incidents**  
   - Only store/use embeddings from real GNN inference. If no GNN ran, don’t write a synthetic embedding; either omit the row or mark “no_embedding” so the UI can show “Similar incidents unavailable (model not run).”

5. **Continual improvement**  
   - Implement the feedback loop: periodic job that calls `load_feedback_batch`, runs real finetune (or threshold/calibration) using (embedding, label), and saves a new checkpoint or calibration so the next run uses updated behavior.

6. **Demo / delete test**  
   - After the above: removing the GNN should visibly degrade or disable detection, watchlists, similar incidents, and explanations — so the demo *depends* on the model.

---

## Summary

| Surface | GNN-driven today? | Change to be GNN-driven |
|--------|--------------------|--------------------------|
| Detection | Partial (optional blend) | Require or clearly gate on model |
| Explanations | No (stub subgraph) | Wire real GNN explainer |
| Watchlists | No (score + hashes) | Add embedding-centroid watchlist |
| Similar incidents | Only if GNN ran (fallback is fake) | No synthetic embeddings; surface “no model” when needed |
| Continual improvement | No (stub) | Implement finetune/calibration from feedback |

Once these are in place, the graph representation and GNN outputs become the actual substrate: detection, investigation, watchlists, similar incidents, and improvement all depend on the model, and the “delete the GNN” test fails (demo breaks or clearly degrades).
