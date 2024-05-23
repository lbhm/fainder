# Explanation of the LLM predicate workload

As of April 25, 2024, the following LLMs lead the Chatbot Arena ranking:

| Rank Model | Arena                           | Elo  |
|------------|---------------------------------|------|
| 1          | GPT-4-Turbo-2024-04-09          | 1258 |
| 2          | GPT-4-1106-preview              | 1253 |
| 3          | Claude 3 Opus                   | 1251 |
| 4          | Gemini 1.5 Pro API-0409-Preview | 1249 |
| 5          | GPT-4-0125-preview              | 1248 |
| 6          | Meta Llama 3 70b Instruct       | 1213 |
| 7          | Bard (Gemini Pro)               | 1208 |
| 8          | Claude 3 Sonnet                 | 1201 |
| 9          | Command R+                      | 1192 |
| 10         | GPT-4-0314                      | 1188 |

Out of this leaderboard, the following models are not available on Chatbot Arena via the direct chat functionality:

- GPT-4-0125-preview
- Bard (Gemini Pro)
- GPT-4-0314

Consequently, we oversampled the available models from the same organization and requested:

- 180 samples from GPT-4-Turbo-2024-04-09
- 120 samples from Gemini 1.5 Pro API-0409-Preview

The raw answers from each model are available under `raw_responses/`.
We manually harmonized the formatting of each response and compiled them into a single list in `responses.txt`.
This list is consequently read by `parse_responses.py`, converted into `PercentilePredicate` instances, and written to `llm_queries.zst`.
If a model produced more than 50 unique and valid percentile predicate instances, we sample 50 examples from the collection.
Otherwise, we include all valid and unique predicates in `llm_queries.zst`.
