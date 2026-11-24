# tau2-bridge

tau2-bench (sierra-research/tau2-bench v1.0.1, commit fc0055dc4e0a, MIT) driven by OUR agent loop.
`src/` is the pinned clone, `.venv/` is its own venv (tau2 + gymnasium + CPU torch + transformers) so
nothing here can break `grpo-local/.venv`.

- `run_tau2.py`: our served loop (sandbox-gym `agent.chat_once`, `tools.parse_action`,
  `agent.observation_text`, imported, not copied) against tau2's `AgentGymEnv`. Thinking on, 8192-token
  budget per episode, out of budget = fail. The official tau2 reward, reported as pass^1..pass^k.
- `tasks.json`: the frozen subset, 5 test-split tasks per domain (seed 20261002). Every model runs the same ids.
- `test_tau2_bridge.py`: stub tests (scripted agent + scripted user on tau2's mock domain, no server, no API).

## Rules

- **EVAL-ONLY.** No tau2 task (any domain, any split) ever goes into an SFT or GRPO pool, a collection
  run, or a prompt/hint. `sft/collect.py` only generates from our own gyms, so nothing feeds it
  tau2 data. Keep it that way. If a tau2 domain ever becomes a training target, split it first and
  seal a test portion that training never sees.
- **One user simulator.** The simulated user is a second LLM (default `deepseek/deepseek-flash`,
  temperature 0, key in `DEEPSEEK_API_KEY`). It is recorded in every result file, and `--compare` refuses runs
  whose sim model or temperature, judge, tau2 version, task ids, budget or trial count differ.
- **pass^k, never single attempts.** The default is `--trials 4`.
- **A request error is never a failure.** A vLLM error, or a tau2 user_error / infrastructure_error /
  orchestrator crash, is retried and then recorded as `missing`, and excluded from pass^k.
- **Graders.** airline (DB + COMMUNICATE) and telecom (ENV_ASSERTION, ACTION) are mechanical.
  39/40 retail test tasks need NL_ASSERTION (an LLM judge), so retail only runs with `--judge-llm`,
  which is recorded.

## Run (home GPU free, vLLM on :8000 serving base + r3)

```powershell
$env:DEEPSEEK_API_KEY = "<key>"
.venv\Scripts\python.exe run_tau2.py --tag base4b --served-model Qwen/Qwen3.5-4B --trials 4
.venv\Scripts\python.exe run_tau2.py --tag r3     --served-model r3              --trials 4
.venv\Scripts\python.exe run_tau2.py --compare results\base4b.summary.json results\r3.summary.json
```
