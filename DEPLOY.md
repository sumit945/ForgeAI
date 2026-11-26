# Deploying the ForgeAI console as a website

ForgeAI builds and chats with agents through a local Ollama model, and free web hosts do not run Ollama. The hosted site is therefore a
**read-only demo**: it shows the real builds committed in `runs/` (dashboard, projects, scores, eval sets, optimizer history, honest reports and
the recorded answers of the Nimbus test agent). Building and live chat are switched off with a clear message.

## Streamlit Community Cloud (free)

1. Push this repository to GitHub (already at `sumit945/forgeai`, branch `main`).
2. Go to https://share.streamlit.io and sign in with GitHub, then **Create app** -> *Deploy a public app from GitHub*.
3. Repository `sumit945/forgeai`, branch `main`, main file path `forgeai/ui.py`.
4. **Advanced settings**: Python **3.12** or newer; in *Secrets* add the single line `demo = true`.
5. Deploy. The first start installs `requirements.txt` (about 2-4 minutes).

Without the `demo = true` secret the console still works: it detects that no model server is reachable and switches building and chat
off with the message "The model server is not reachable".

## Full app (live builds and chat)

Run it where Ollama runs: a machine or cloud VM with `ollama pull qwen2.5:3b-instruct` (and `bge-m3` for hybrid retrieval), then
`python -m forgeai ui`. A GPU makes builds several times faster. Set `FORGEAI_WORKERS` together with `OLLAMA_NUM_PARALLEL` to run more model calls at once.
Do not expose an unauthenticated instance to the internet: use `--api-key` for the REST API and put the console behind a login or VPN.
