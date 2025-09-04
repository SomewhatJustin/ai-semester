# Week 1 — Orientation & Systems View
**Read:** Ch.1 (Intro & AI stack)
**Lab:** Stand up both:
(a) API model call (Claude/GPT)
(b) Local inferencing (Llama via vLLM or Ollama). Compare latency/cost qualitatively.
**Deliverable:** Repo scaffold and README.

# Week 2 — Foundation Models Deep Dive
**Read:** Ch.2 (Data → Architecture → Post-training; Sampling)
**Lab:** Implement a small script that sweeps decoding params (temperature, top-p, top-k) and logs output diversity + length.
**Deliverable:** Short blog note: “What decoding knobs actually did on Task X.”

# Week 3 — Evaluation Fundamentals
**Read:** Ch.3 (metrics, AI-as-judge, comparative eval)
**Lab:** Build `eval/run.py` that:
- runs a set of prompts across 2 models,
- computes exact-match / simple similarity,
- supports an “LLM judge” rubric prompt.
**Deliverable:** Eval harness v0 + 10–20 sample cases under version control.

# Week 4 — System-Level Evaluation
**Read:** Ch.4 (criteria, model selection, pipelines)
**Build 1 (start):** LLM Evaluator Dashboard
Minimal web UI to load a prompt set, run models A/B, visualize accuracy/latency/$, and export CSV.
**Milestone:** Public update #1 with screenshots + numbers.

# Week 5 — Prompt Engineering in Practice
**Read:** Ch.5 (prompt best practices, context efficiency, defensive prompting)
**Lab:** Add prompt templates + versioning; create a regression test that fails when a prompt change degrades metrics by >X%.
**Deliverable:** Evaluator Dashboard v1 with prompt version pins + regression guard.

# Week 6 — RAG (Retrieval-Augmented Generation)
**Read:** Ch.6 (RAG architecture, retrieval optimization)
**Build 2 (start):** Personal Notes RAG
Ingest markdown/PDFs, build embeddings index, rerank top-k, cite spans.
**Baseline eval set:** 25 Q/A from your own notes.
**Milestone:** 50-question eval set with accuracy ≥ baseline.

# Week 7 — Agents & Tools
**Read:** Ch.6 (Agents: tools, planning, failure modes)
**Lab:** Add 2 tools (e.g., web search + code exec sandbox or calendar/notes).
**Deliverable:** Agent reliability table (success rate, timeouts, failure taxonomy).

# Week 8 — Midterm Practical (Timed)
**2-hour build:** Given a new mini-dataset + task spec, ship a working prompt/RAG pipeline and report metrics.
**Deliverable:** Midterm report (repo tag v0.5), reflection on trade-offs.

# Week 9 — When (Not) to Finetune
**Read:** Ch.7 (finetune rationale, memory math, PEFT/LoRA/QLoRA)
**Lab:** PEFT on a 7B model with a tiny domain dataset (e.g., 2–5k pairs). Track loss curves + eval vs. non-tuned baseline.
**Deliverable:** Finetune report: cost, hours, GPU, delta-accuracy, drift risks.

# Week 10 — Dataset Engineering
**Read:** Ch.8 (curation, dedup, filtering, synthesis, distillation)
**Lab:** Build a data pipeline: dedup, quality filters, stratified splits. Optional: synthesize augmentations with constrained prompts.
**Deliverable:** `/data/pipeline/Makefile` + data report with charts.

# Week 11 — Inference Optimization I (Model)
**Read:** Ch.9 (accelerators, perf metrics, quantization)
**Lab:** Compare fp16 vs. 4-bit quant, batch size, and max tokens on latency/throughput.
**Deliverable:** Perf notebook + a “prod defaults” JSON checked in.

# Week 12 — Inference Optimization II (Service)
**Read:** Ch.9 (service optimization)
**Lab:** Deploy vLLM/TGI behind FastAPI with request batching, caching, and timeouts.
**Deliverable:** k6 (or Locust) load test results + flamegraph/s.

# Week 13 — Architecture & Guardrails
**Read:** Ch.10 (context enrichment, guardrails, router/gateway, caches, orchestration)
**Build 3 (start):** AI Product Skeleton
End-to-end app using: router (choose model by task), guardrails (JSON schema/regex), caching layers, and monitoring hooks.
**Milestone:** Public update #2 with an architecture diagram and perf numbers.

# Week 14 — User Feedback & Monitoring
**Read:** Ch.10 (feedback loops, observability)
**Lab:** Add thumbs-up/down + free-text feedback; wire to retraining/eval queue.
**Deliverable:** Close the loop: a cron/workflow that re-runs eval on fresh data weekly.

# Week 15 — Capstone Build & Polish
**Capstone Options (pick one):**
- LLM Analytics Suite (Evaluator + RAG + Router + Metrics) as a single cohesive product.
- Specialist Agent for a real workflow (e.g., engineering triage, support, or research assistant) with tools and guardrails.
- Mini-model: small PEFT-tuned specialist with a reproducible data pipeline and a clean serving stack.
**Deliverables:** Feature-complete, docs, Dockerized.

# Week 16 — Capstone Demo & Final Practical
**Practical (90 min):** Given a failure ticket (e.g., latency spike + accuracy drop), diagnose and fix; produce a postmortem.
**Capstone Demo:** 5-min video, README with benchmark table, live URL or screencast, and a blog post.
