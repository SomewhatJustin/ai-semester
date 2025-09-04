# WEEK 1
I am going to build a simple tool for comparing local models to hosted models. See [bench.py](./tools/bench.py).

I did a simple benchmark between gpt-oss:20b running locally via Ollama vs GPT-5. Given the simple nature of my prompts, I'm not surprised to see that the outputs are roughly similar in quality. I am also not surprised that Ollama was slower in getting responses. I'm working on an NVIDIA RTX 4060 with a strong wired ethernet connection.

It would be interresting to test an additional, much smaller model, somewhere around 7B parameters to see if you could get comparable results but with better speeds.

```
2025-09-03 14:46:09,051 - INFO - BENCHMARK SUMMARY
2025-09-03 14:46:09,051 - INFO - ============================================================
2025-09-03 14:46:09,051 - INFO - Total prompts: 10
2025-09-03 14:46:09,051 - INFO - OpenAI (gpt-5-2025-08-07): 10/10 successful, avg 4.74s
2025-09-03 14:46:09,051 - INFO - Ollama (gpt-oss:20b): 10/10 successful, avg 12.09s
```

UPDATE: I am now testing with gemma3:4b. Saying hello to it in the `ollama` CLI already demonstrates to me just how much faster it is than `gpt-oss:20b` - so I'm curious if it'll be smart enough.

```
============================================================
2025-09-03 15:19:39,047 - INFO - BENCHMARK SUMMARY
2025-09-03 15:19:39,047 - INFO - ============================================================
2025-09-03 15:19:39,047 - INFO - Total prompts: 10
2025-09-03 15:19:39,047 - INFO - OpenAI (gpt-5-2025-08-07): 10/10 successful, avg 4.96s
2025-09-03 15:19:39,047 - INFO - Ollama (gemma3:4b): 10/10 successful, avg 1.60s
2025-09-03 15:19:39,047 - INFO - ============================================================
```

Much faster! Even faster than API calls.

But not without a cost.

One of my prompts is

> If all bloops are bloopsies and some bloopsies are glips, are some bloops necessarily glips? Answer yes/no with 1 sentence.

The answer should be "No". GPT-5 and gpt-oss:20b both got it right every time. From the last run of GPT-5:

> No; the bloopsies that are glips need not include any bloops.

Gemma3:4b, on the other hand:

> Yes, some bloops are necessarily glips because if all bloops are bloopsies and some bloopsies are glips, then the bloops that are also bloopsies must be glips.

The reasoning sounds ~right but falls down upon further inspection.

You can see more in the [results csv](../data/results/benchmark_results_20250903_151939.csv)

**9/4/25**
Added cost for API calls. GPT-5 is quite cheap!
