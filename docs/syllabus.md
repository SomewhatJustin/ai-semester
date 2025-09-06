# Personal Semester: AI Engineering — 16-Week Schedule

**Primary text:** Chip Huyen, *AI Engineering*
**Focus:** ML Research Engineer + AI Product Engineer
**Weekly load:** ~8–12 hrs

---

# Week 1 — Orientation & Systems View
**Read:** Ch.1 (Intro & AI stack)
**Lab:** Stand up both:
(a) API model call (Claude/GPT)
(b) Local inferencing (Llama via vLLM or Ollama). Compare latency/cost qualitatively.
**Deliverable:** Repo scaffold and README.

## Week 2 — Foundation Models & Sampling
**Read:** Ch.2 (data → architecture → post-training; sampling)
**Lab:** Decoding sweep (temperature/top-p/top-k/mirostat); log diversity/length/TTFT
**Deliverable:** Short note: “How decoding knobs changed outcomes on Task X”

## Week 3 — Evaluation Methodology (Metrics & Judges)
**Read:** Ch.3 (perplexity/cross-entropy; exact vs. AI-judge; comparative eval)
**Lab:** `eval/run.py` v0: run prompts across 2 models; exact-match + embedding sim + AI-judge rubric
**Deliverable:** 20-case eval set + reproducible script

## Week 4 — System-Level Evaluation & Model Selection
**Read:** Ch.4 (criteria, model selection workflow, build vs. buy, private leaderboards)
**Build:** **Evaluator Dashboard v1** — A/B models, accuracy/latency/$ charts, CSV export
**Deliverable:** Public Update #1 with screenshots + metrics

## Week 5 — Prompt Engineering & Hardening
**Read:** Ch.5 (prompt anatomy, context efficiency, defensive prompting)
**Lab:** Prompt templates + versioning; regression tests that fail on >X% metric drop
**Deliverable:** Prompt registry + CI check; “prompt diffs” report

## Week 6 — RAG Foundations
**Read:** Ch.6 (RAG architecture; BM25 vs. embeddings; retriever quality)
**Build:** **Notes RAG v1** — ingest MD/PDF; FAISS/Chroma; top-k + citations
**Deliverable:** 50-question private eval set; baseline accuracy report

## Week 7 — Agents, Tools, & Failure Modes
**Read:** Ch.6 (tools, planning, reflection/memory, failure taxonomy)
**Lab:** Add 2 tools (e.g., web search + code exec or calendar/notes); log success/failure taxonomy
**Deliverable:** Agent reliability table (success %, timeout %, jailbreak/hijack notes)

## Week 8 — Midterm Practical (Timed)
**Practical (2h):** Given a mini-dataset + spec, ship a working prompt/RAG pipeline with numbers
**Deliverable:** Midterm tag `v0.5`; postmortem (what worked, what failed, next fixes)

## Week 9 — When (Not) to Finetune
**Read:** Ch.7 (reasons to/NOT to finetune; memory math; PEFT/LoRA/QLoRA; merging)
**Lab:** PEFT on 7B with small domain set (2–5k pairs); track loss + task deltas vs. RAG
**Deliverable:** Finetune report (quality delta, GPU hrs, cost, serve plan). Use Prime Intellect to rent GPUs.

## Week 10 — Dataset Engineering Pipeline
**Read:** Ch.8 (quality/coverage/quantity; synth data; verification; distillation)
**Lab:** Data pipeline: dedup, filters, splits; (opt) synth augmentations with guardrails + eval
**Deliverable:** `/data/pipeline` + charts and quality gates

## Week 11 — Inference Optimization I (Model)
**Read:** Ch.9 (metrics: TTFT/TPOT/throughput; quantization; attention/KV cache basics)
**Lab:** Compare fp16 vs. 4-bit; batch sizes; max tokens; KV cache impact on long contexts
**Deliverable:** Perf notebook + “prod defaults” JSON

## Week 12 — Inference Optimization II (Service)
**Read:** Ch.9 (batching, parallelism, prompt caching, decoupled prefill/decoding)
**Build:** **Serve vLLM/TGI behind FastAPI** with request batching + caching; k6/Locust load test
**Deliverable:** p50/p95 latency, throughput curves, and flamegraph

## Week 13 — Architecture, Guardrails, & Routing
**Read:** Ch.10 (context enrichment; guardrails; model router/gateway; caches; orchestration)
**Build:** **AI Product Skeleton** — router (task → model), JSON-schema guardrails, caches, obs hooks
**Deliverable:** Public Update #2 with high-level architecture + SLOs

## Week 14 — User Feedback & Data Flywheel
**Read:** Ch.10 (conversational feedback, analytics, flywheels, observability)
**Lab:** Thumbs up/down + free-text; feedback → eval queue; weekly cron to re-run evals
**Deliverable:** Feedback → retraining/eval loop diagram + working job

## Week 15 — Capstone Build & Polish
**Pick ONE Capstone:**
1) **LLM Analytics Suite** (Evaluator + RAG + Router + Metrics)
2) **Specialist Agent** for a real workflow with tools + guardrails
3) **Mini-model** (QLoRA specialist) with reproducible data/serve stack
**Deliverable:** Feature-complete; Dockerized; docs + demo script

## Week 16 — Final Practical + Demo
**Practical (90m):** Diagnose & fix a staged incident (latency spike + accuracy drop); postmortem
**Demo:** 5-min video; README with benchmark table; live URL or screencast; blog post
**Deliverable:** Resume bullets + “What I’d do next” section

---

### Ongoing (every week)
- Update README learning log
- Track metrics (accuracy/latency/$) for any change
- Maintain issues/milestones; push bi-weekly public updates (Weeks 4 & 13)

### Optional OSS Track (Weeks 11–13)
- Triage → small PR (docs/tests/bug) in vLLM/llama.cpp/Triton/HF
- Stretch: perf or integration PR + short write-up
