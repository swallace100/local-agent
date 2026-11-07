# Local AI Agent

A private agent that runs on your own GPU. It can research on the web, write articles, and work on code.

It combines [OpenCode](https://opencode.ai) (the agent), [Ollama](https://ollama.com) with one local model, and a self-hosted [SearXNG](https://docs.searxng.org) instance for web search. There are no API costs.

## Setup

You need Docker Desktop, Node.js, and Ollama (`winget install Ollama.Ollama`). Start Docker Desktop, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

The script downloads the model (about 6.6 GB), starts SearXNG, and installs the agent's config so that it works in any folder. It's safe to run again whenever you change the config.

## Usage

Open a terminal in any project folder and run `opencode`. Press **Tab** to switch modes:

| Mode | Use it for | Can run commands? |
|---|---|---|
| `build` | Writing and fixing code, with web access for looking things up | Asks you first |
| `plan` | Discussing an approach without changing files | Asks you first |
| `research` | A cited report on a question | No |
| `writer` | A researched, self-edited article, saved to `articles/` | No |

To run a single task without the interface:

```powershell
opencode run --agent research "Compare SQLite and DuckDB for local analytics"
```

After a restart, make sure Docker Desktop and Ollama are running. If searches fail, run `docker compose up -d` from this repo.

## What to expect

Measured on an RTX 4070 Laptop (8 GB) with 31 GB of RAM:

| Task | Time |
|---|---|
| Short research question, including the model loading from cold | about 2 minutes |
| Fix two bugs, add a function and tests, and run them | about 5 minutes |

The model handles research summaries and small, well-defined coding tasks. Expect it to struggle with large changes across many files, and check its citations and code.

## Safety

- Commands run directly on your machine, not in a sandbox. `build` asks before each one, so read them before approving.
- `research` and `writer` read untrusted web pages, so they can't run commands. They can still edit files in the current project, so use version control.
- `build`, `research` and `writer` can edit files in the current project without asking. `plan` can't edit files.
- Access outside the current project asks first in `build` and `plan`, and is blocked in `research` and `writer`.
- `.env` files are blocked from the file-reading tool. A command you approve could still read them.
- `web_read_page` refuses local and private network addresses.
- Search queries and page requests still reach the internet, though the model and your files stay local.

## Customizing

| To change | Edit |
|---|---|
| Permissions, default model | [opencode.jsonc](opencode.jsonc) |
| A mode's instructions, or add a new mode | [.opencode/agents/](.opencode/agents/) |
| Model or context size | [ollama/Modelfile](ollama/Modelfile), and keep `limit.context` in `opencode.jsonc` in sync |
| Search and page-reading tools | [tools/web-search/](tools/web-search/) (`npm test` runs its tests) |
| Search engine settings | [searxng/settings.yml](searxng/settings.yml) |

Run the setup script afterwards so your changes apply in other folders too.

A larger context fits more but runs slower, because more of the model spills from the GPU onto the CPU:

| `num_ctx` | CPU/GPU split | Speed |
|---|---|---|
| 8K | 12% / 88% | 35 tokens/s |
| 16K | 16% / 84% | 30 tokens/s |
| 24K | 21% / 79% | 26 tokens/s |
| 32K (current) | 24% / 76% | 23 tokens/s |

## License

MIT. See [LICENSE](LICENSE).
