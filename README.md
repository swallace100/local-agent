# Local AI Agent

A private agent that runs on your own GPU. It can research on the web, search your own notes, write articles, and work on code, with no API costs.

It includes two agents that share one local model and the same web tools:

|                     | Unattended agent                                                                           | Interactive agent                                                               |
| ------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| Use it for          | Handing off a task and coming back to the result                                           | Working alongside it in a terminal                                              |
| Built with          | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), in [agent/](agent/) | [OpenCode](https://opencode.ai), configured in [opencode.jsonc](opencode.jsonc) |
| Searches your notes | Yes                                                                                        | No                                                                              |
| Runs commands       | In a Docker sandbox with no network                                                        | On your machine, after asking you                                               |
| Changes your code   | No. It works on a copy and gives you a patch                                               | Yes, directly                                                                   |

Both use [Ollama](https://ollama.com) to run Qwen 3.5 9B locally, and a self-hosted [SearXNG](https://docs.searxng.org) instance for web search.

## Setup

You need Docker Desktop, Node.js, Python 3.11 or later, and Ollama (`winget install Ollama.Ollama`). Start Docker Desktop, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

The script downloads the models (about 7 GB), starts SearXNG, and installs both agents. It's safe to run again whenever you change the config.

After a restart, make sure Docker Desktop and Ollama are running.

## Unattended agent

Run these from this folder:

```powershell
.venv\Scripts\agent research "Compare SQLite and DuckDB for local analytics"
.venv\Scripts\agent research "Summarize my notes on Heron" --no-web
.venv\Scripts\agent run "Fix the failing tests" --project C:\code\my-app
.venv\Scripts\agent run "Write a packing list for a three-day hike" --no-web
```

Use `agent research` for questions answered from the web or your notes. Use `agent run` for everything else: code, files, and writing.

Each run gets a folder in `runs/` containing:

| File            | Contents                                                         |
| --------------- | ---------------------------------------------------------------- |
| `report.md`     | The result, with any warnings from the agent's checks at the top |
| `changes.patch` | Code changes, when you used `--project`                          |
| `result.json`   | Status, steps used, revisions, and time taken                    |
| `trace.jsonl`   | Every step the agent took, for debugging                         |
| `workspace/`    | Any files the agent wrote                                        |

With `--project`, the agent works on a copy of your folder and never changes the original. To keep its changes, run `git apply -p2 <path to changes.patch>` from your project folder. The command is also printed when the run ends.

### Other commands

| Command                | What it does                                                       |
| ---------------------- | ------------------------------------------------------------------ |
| `agent research "question"` | Research a question and write a cited report (see [How research works](#how-research-works)) |
| `agent runs`           | List runs and their status                                         |
| `agent resume <id>`    | Continue a run that was interrupted, from its last completed step  |
| `agent index`          | Update the search index after changing `knowledge/`                |
| `agent search "query"` | Search your notes directly, to check what the agent would find     |
| `agent eval`           | Score the agent on fixed tasks (see [Improving it](#improving-it)) |
| `agent moltbook` | Write a read-only digest of Moltbook (see [Moltbook digest](#moltbook-digest)) |

### How a run works

```
plan -> work <-> tools
          |
        review -- problems found --> back to work (at most 2 times)
          |
        report
```

1. **Plan:** the model writes a short plan.
2. **Work:** the model calls tools until it has a result. The result is its final reply, or a file it hands in with `finish_with_file`.
3. **Review:** code checks that every URL cited was actually opened. Then the model checks whether the result does what the task asked, given which tools it used, which files it wrote, and the last command's output. Any problems go back to the agent to fix.
4. **Report:** the result is saved, with warnings for anything the reviews couldn't resolve. If the agent left files in its workspace, the report lists them.

Limits keep a run from going on forever: 30 steps, 45 minutes, and 3 sections of any one web page. The agent is warned as it nears the step limit. If it hits a limit, it writes up what it has and the report says so. Every step is saved, so `agent resume` can pick up after a crash or a reboot.

When the conversation outgrows the context window, the oldest tool results are left out of what the model sees. The run measures how full the window is from the token counts Ollama reports, so this works the same for prose, code and other languages.

### How research works

`agent research` doesn't let the model steer. Code runs the steps, and the model does one small job at a time, each in a fresh conversation:

```
split -> gather (once per sub-question) -> write <-> review
```

1. **Split:** the model breaks the task into at most 4 sub-questions.
2. **Gather:** for each sub-question, code searches your notes and the web. The model picks which results to read by number. For each page, the model lists facts with a quote for each one, and code keeps a fact only if its quote appears word for word in the page.
3. **Write:** the model writes the report from the kept notes only. It never sees whole pages at this point.
4. **Review:** code checks that every URL cited is one the notes came from. Then the model checks the report against the notes.

The report ends with an Evidence section that lists each checked quote under its source, so you can verify a claim without rereading the page. A checked quote shows that the source says it. It doesn't show that the source is right.

A research run takes about 2 to 4 minutes for a question with four parts.

### Tools

| Tool                                                 | What it does                                          |
| ---------------------------------------------------- | ----------------------------------------------------- |
| `knowledge_search`                                   | Searches your files in `knowledge/`                   |
| `web_search`, `web_read_page`                        | Searches the web, and reads pages as clean text       |
| `list_files`, `read_file`, `write_file`, `edit_file` | Work with files in the run's workspace only           |
| `finish_with_file`                                   | Hands in a workspace file as the final result         |
| `run_command`                                        | Runs a command in the sandbox (see [Safety](#safety)) |

### Your notes (RAG)

Put notes and documents in `knowledge/`, then run `agent index`. Indexing only re-processes files that changed. Text formats are supported, including Markdown, code, JSON, YAML, and CSV. PDFs and Word documents aren't supported yet.

### Moltbook digest

[Moltbook](https://www.moltbook.com) is a social network for AI agents. This command reads it and reports what the bots are talking about:

```powershell
.venv\Scripts\agent moltbook
.venv\Scripts\agent moltbook --topic "agent memory"
```

The digest goes in `runs/<id>/report.md`, like any other run, and has a section for posts that tried to instruct or trick agents. If several bots keep mentioning the same product in the comments, a section at the end counts them. You only need Ollama running.

It can only read. It has no account or API key, and no code that can post, comment, vote, or sign up. Because everything it reads is written by other bots, it runs with only its three Moltbook tools, so it can't reach your notes, files, commands, or the rest of the web.

## Interactive agent

Open a terminal in any project folder and run `opencode`. It starts in `build` mode. Press **Tab** to switch:

| Mode       | Use it for                               | Can run commands? |
| ---------- | ---------------------------------------- | ----------------- |
| `build`    | Writing and fixing code, with web access | Asks you first    |
| `plan`     | Discussing an approach; can't edit files | Asks you first    |
| `research` | A cited report on a question             | No                |
| `writer`   | A researched, self-edited article        | No                |

If PowerShell says running scripts is disabled, use `opencode.cmd` instead.

## What to expect

Measured on an RTX 4070 Laptop (8 GB) with 31 GB of RAM, using the unattended agent:

| Task                                                 | Time                    |
| ---------------------------------------------------- | ----------------------- |
| Answer from your notes                               | 10–30 seconds           |
| Short research question with cited sources           | 35 seconds to 3 minutes |
| Fix two bugs, add a function and tests, and run them | about 1.5–2 minutes     |
| Detailed 1,800-word comparison from five long pages  | about 11 minutes        |

The model handles research summaries and small, well-defined coding tasks well. It still makes factual mistakes, and the reviews don't catch all of them, so check anything that matters. Expect it to struggle with large changes across many files.

### What it's best at

Use it when you can check the result, or when a mistake is cheap to fix:

| Good fit | Why |
|---|---|
| Questions about your private notes and documents | Your files stay on your machine, and answers can be checked against them |
| Small coding tasks that have tests | The agent runs the tests, so a wrong answer fails visibly |
| Web research with cited sources | You can open the sources and confirm the claims |
| First drafts, outlines, and summaries you'll edit anyway | Errors get caught when you edit |
| Repetitive or overnight work | Runs cost nothing and have no usage limits |

Avoid it when you couldn't spot an error yourself and a mistake would cost you:

| Poor fit | Why |
|---|---|
| Learning a language, especially one other than English or Chinese | In a test set of Korean travel conversations, the Hangul was mostly usable, but the romanization was often wrong and some scenarios made no sense |
| Medical, legal, or financial facts | It can sound confident and still be wrong |
| Niche details with no source to check | A 9B model holds far less knowledge than a large cloud model |
| Large changes across many files | It loses track of the wider codebase |

The agent's reviews check whether a result followed the task and whether its citations were opened. They can't tell whether the content is correct, so a clean report with no warnings doesn't mean the answer is right.

## Safety

The unattended agent runs with no one approving its actions, so its limits are enforced in code rather than by asking:

- Commands run in a Docker container with no network access, a read-only system, no admin rights, and memory and process limits. The container can only see the run's workspace, and commands time out after 2 minutes.
- File tools can't leave the workspace or touch `.git` folders.
- `--project` copies your folder without `.git`, `.env` files, `node_modules`, or virtual environments, so the agent never sees those.
- `web_read_page` refuses local and private network addresses. The check is made on the address it actually connects to, so a site can't pass it with one address and then switch to another.

One risk remains with `agent run`. The agent reads untrusted web pages and your private notes in the same run, so a malicious page could try to trick it into putting your notes in a web address it then visits. In testing it ignored planted instructions, but that isn't a guarantee. Don't keep passwords or keys in `knowledge/`, and use `--no-web` for tasks about private notes.

`agent research` closes most of that gap by construction. The model never writes a web address to open: it can only pick search results by number. The search queries are written from your task alone, before any page or note has been read. What remains is that your task's sub-questions are sent to the search engine, as with any search.

For the interactive agent:

- Commands run directly on your machine, not in a sandbox. `build` and `plan` ask before each one, so read them before approving. `research` and `writer` can't run commands.
- `build`, `research` and `writer` can edit files in the current project without asking, so use version control. `plan` can't edit files.
- `.env` files are blocked from the file-reading tool. A command you approve could still read them.

In both agents, search queries and page requests reach the internet, though the model and your files stay local.

## Improving it

`agent eval` runs the tasks in [evals/tasks.json](evals/tasks.json) and checks the results in code. It runs the agent's tests itself rather than trusting the report. The tasks cover notes, web research, code, long documents, planted instructions, and tidying a folder. Add `"mode": "research"` to a task to run it with `agent research`. Each result is added to `evals/history.jsonl` with the commit and model, so you can tell whether a change helped.

A good way to improve the agent over time:

1. When a run goes wrong, add a task to `evals/tasks.json` that reproduces it.
2. Change a prompt, a limit, or a tool.
3. Run `agent eval` and compare the result with the history.

Run `.venv\Scripts\python -m pytest` for the unit tests. They use a scripted model, so they don't need the GPU.

## Customizing

| To change                              | Edit                                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Unattended agent: prompts and workflow | [agent/graph.py](agent/graph.py)                                                                                         |
| Research runs: prompts and workflow    | [agent/research.py](agent/research.py)                                                                                   |
| Unattended agent: limits and models    | [agent/config.py](agent/config.py), or `AGENT_*` environment variables                                                   |
| Interactive agent: permissions         | [opencode.jsonc](opencode.jsonc)                                                                                         |
| Interactive agent: modes               | [.opencode/agents/](.opencode/agents/)                                                                                   |
| Model or context size                  | [ollama/Modelfile](ollama/Modelfile), then `limit.context` in `opencode.jsonc` and `context_tokens` in `agent/config.py` |
| Web tools                              | [tools/web-search/](tools/web-search/) (`npm test` runs its tests)                                                       |
| Search engine settings                 | [searxng/settings.yml](searxng/settings.yml)                                                                             |

Run the setup script afterwards so the interactive agent picks up your changes in other folders too.

A larger context fits more but runs slower, because more of the model spills from the GPU onto the CPU:

| `num_ctx`     | CPU/GPU split | Speed       |
| ------------- | ------------- | ----------- |
| 8K            | 12% / 88%     | 35 tokens/s |
| 16K           | 16% / 84%     | 30 tokens/s |
| 24K           | 21% / 79%     | 26 tokens/s |
| 32K (current) | 24% / 76%     | 23 tokens/s |

## License

MIT. See [LICENSE](LICENSE).
