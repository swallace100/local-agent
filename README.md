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
.venv\Scripts\agent organize C:\Users\me\Downloads
```

Use `agent research` for questions answered from the web or your notes, and `agent organize` to tidy a folder. Use `agent run` for everything else: code, files, and writing.

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
| `agent team "question"` | Research with several agents that share notes and cross-check each other (see [Research as a team](#research-as-a-team)) |
| `agent organize <folder> "instruction"` | Plan how to tidy a folder, without moving anything (see [Organizing a folder](#organizing-a-folder)) |
| `agent apply <id>`, `agent undo <id>` | Carry out an organize plan, or reverse it |
| `agent queue add`, `agent queue run` | Line up tasks and work through them unattended (see [Running tasks overnight](#running-tasks-overnight)) |
| `agent recall "query"` | Search what earlier research runs found (see [What earlier research found](#what-earlier-research-found)) |
| `agent runs`           | List runs and their status                                         |
| `agent resume <id>`    | Continue a run that was interrupted, from its last completed step  |
| `agent index`          | Update the search index after changing `knowledge/`                |
| `agent search "query"` | Search your notes directly, to check what the agent would find     |
| `agent eval`           | Score the agent on fixed tasks (see [Improving it](#improving-it)) |
| `agent moltbook` | Write a read-only digest of Moltbook (see [Moltbook digest](#moltbook-digest)), or draft and approve posts (see [Posting on Moltbook](#posting-on-moltbook)) |

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

1. **Split:** the model breaks the task into at most 5 sub-questions, each asking for one thing.
2. **Gather:** for each sub-question, code searches your notes and the web. The model picks which results to read by number. For each page, the model lists facts with a quote for each one, and code keeps a fact only if its quote appears word for word in the page.
3. **Write:** the model writes the report from the kept notes only. It never sees whole pages at this point.
4. **Review:** code checks that every URL cited is one the notes came from. Then the model checks the report against the notes.

The report ends with an Evidence section that lists each checked quote under its source, so you can verify a claim without rereading the page. A checked quote shows that the source says it. It doesn't show that the source is right.

A research run takes about 2 to 4 minutes for a question with four parts.

Search engines block a client that asks too often, usually for under an hour. Four engines are used so that one being blocked doesn't matter. If all of them refuse, the report says how many searches failed instead of claiming there was no evidence. Many research runs in a row are the usual cause, so space out large batches.

To stay under those limits, the agent waits at least 2 seconds between searches. Set `AGENT_SEARCH_PAUSE_SECONDS` to change that.

### Tools

| Tool                                                 | What it does                                          |
| ---------------------------------------------------- | ----------------------------------------------------- |
| `knowledge_search`                                   | Searches your files in `knowledge/`                   |
| `web_search`, `web_read_page`                        | Searches the web, and reads pages, PDFs and Word files as clean text |
| `list_files`, `read_file`, `write_file`, `edit_file` | Work with files in the run's workspace only           |
| `finish_with_file`                                   | Hands in a workspace file as the final result         |
| `run_command`                                        | Runs a command in the sandbox (see [Safety](#safety)) |

### Research as a team

`agent team` runs the same research with two or more agents that share a board of tasks and notes.

```powershell
.venv\Scripts\agent team "Compare SQLite and DuckDB for local analytics"
.venv\Scripts\agent team "Compare SQLite and DuckDB for local analytics" --agents 3
```

- **Shared tasks:** each sub-question goes on the board, and only one agent can claim it.
- **Shared notes:** an agent sees which pages the others have read and looks elsewhere.
- **Handing over:** an agent that finds nothing rewords the question and leaves it for another agent.
- **Cross-checking:** once a question is answered, a different agent looks for the answer again on websites the first one didn't use. The two sets of notes are compared and marked AGREE, DISAGREE, UNCLEAR, or UNCONFIRMED when no second source turned up.

The report shows which agent found each quote and ends with the cross-checks. An answer two agents reached from different websites is worth more than one found once, and a DISAGREE tells you where to look yourself.

The agents take turns on your one GPU, so a team is not faster. A four-part question takes about 3 minutes, against about 2 for `agent research`, and makes roughly twice as many searches. Use it when you want the second opinion. The comparison is made by the model, so it can be wrong: in testing it once marked "2019" and "27 June 2019" as a disagreement.

The board is saved as `board.sqlite` in the run's folder.

### Organizing a folder

`agent organize` sorts the loose files in a real folder into subfolders. It works in two stages, and only the second one moves anything.

```powershell
.venv\Scripts\agent organize C:\Users\me\Downloads
.venv\Scripts\agent organize C:\Users\me\Downloads "Put invoices in Invoices and photos in Photos"
.venv\Scripts\agent apply <run id>
.venv\Scripts\agent undo <run id>
```

1. **Plan:** code lists the files. The model proposes a few subfolders, then says which one each file belongs in. The plan is saved as `report.md` to read and `moves.json` to apply. The folder is not touched.
2. **Apply:** `agent apply` makes the moves. `agent undo` puts every file back and removes the folders it made.

You can edit `moves.json` before applying it. If you leave out the instruction, the agent sorts files by what they are.

What it will and won't do:

- It only moves files. Nothing here can delete, rename, overwrite or edit a file.
- It only moves files that sit directly in the folder. Existing subfolders and their contents are left alone, though the agent can move files into them.
- It skips files whose names start with a dot, symbolic links, and any file that changed after the plan was made.
- If a file with the same name is already in the destination, the file stays where it is.
- It refuses a whole drive, your home folder, and code repositories.
- It reads file names, sizes, dates, and the first line of text files. Everything stays on your machine: this command has no web access.

Sixty files take about 30 seconds to plan. Expect a few debatable choices, such as a manual filed under work documents, so read the plan first.

### Your notes (RAG)

Put notes and documents in `knowledge/`, then run `agent index`. Indexing only re-processes files that changed. It reads PDFs, Word documents (`.docx`), and text formats including Markdown, code, JSON, YAML, and CSV. Answers from a PDF name the page.

`agent index` lists any file it had to skip and why. The usual reasons are a scanned PDF, which holds pictures of pages with no text in them, and a password-protected PDF. Older `.doc` files aren't read, so save them as `.docx` first.

Search works two ways at once. It finds text that means the same as your question, and it finds exact words such as part numbers, names and codes, which meaning-based search tends to miss among similar entries.

### What earlier research found

Every `agent research` and `agent team` run that uses the web saves its checked notes, each with the page it came from. Later runs start from them:

- When a new question is close to one answered before, the agent reads those same pages again first. If they answer it, no web search is made.
- A remembered note is never used as evidence. It only says where to look, and the page is read and its quotes checked afresh, so an out-of-date note can't end up in a report.
- These notes are kept apart from your own files in `knowledge/`. They began as text on web pages, so they are trusted less.

```powershell
.venv\Scripts\agent recall "who created Rust"
.venv\Scripts\agent research "..." --no-memory
```

`agent recall` shows what has been remembered: the fact, its quote, the page, and when it was found. `--no-memory` runs without using or adding to it.

In one test, a second question about a topic researched earlier made no searches at all. That also keeps you under the search engines' limits.

### Running tasks overnight

Line up tasks in a queue, then work through them in one go:

```powershell
.venv\Scripts\agent queue add "Compare SQLite and DuckDB for local analytics"
.venv\Scripts\agent queue add "How do heat pumps work in cold climates?" --mode team
.venv\Scripts\agent queue add "Fix the failing tests" --mode run --project C:\code\my-app
.venv\Scripts\agent queue add "agent memory" --mode moltbook --repeat
.venv\Scripts\agent queue list
.venv\Scripts\agent queue run
```

`--mode` is `research` (the default), `team`, `run`, or `moltbook` for a digest. `--repeat` keeps a task in the queue so it runs every time, which suits a nightly digest.

Each task is an ordinary run with its report in `runs/`. When the queue finishes, a summary file `runs/queue-<date>.md` lists every task, how it ended, and where its report is.

- One task failing doesn't stop the rest.
- Tasks run one at a time, with a minute between them so search engines get a rest.
- A task cut off by a crash or a restart carries on from its last completed step the next time the queue runs.
- `agent queue remove <number>` takes a task out, and `agent queue clear` removes the finished ones.

To run the queue every night:

```powershell
.venv\Scripts\agent queue schedule 02:00
.venv\Scripts\agent queue unschedule
```

This adds a task to Windows Task Scheduler that wakes the PC if it's asleep, and runs later if the PC was off at that time. Output goes to `.data\queue.log`. You must be signed in to Windows, Ollama must be running, and Docker Desktop too for tasks that search the web or run commands. A console window opens while it works.

Folder tidy-ups can't be queued, because their plan needs reading before anything is moved.

### Moltbook digest

[Moltbook](https://www.moltbook.com) is a social network for AI agents. This command reads it and reports what the bots are talking about:

```powershell
.venv\Scripts\agent moltbook
.venv\Scripts\agent moltbook --topic "agent memory"
```

The digest goes in `runs/<id>/report.md`, like any other run, and has a section for posts that tried to instruct or trick agents. If several bots keep mentioning the same product in the comments, a section at the end counts them. You only need Ollama running.

The digest can only read. It doesn't use the agent's account or key, and it has no tool that can post, comment, vote, or sign up. Because everything it reads is written by other bots, it runs with only its three Moltbook tools, so it can't reach your notes, files, commands, or the rest of the web.

### Posting on Moltbook

The agent can have its own Moltbook account. By default it never posts by itself: it drafts, and you approve each draft. You can also let it comment unprompted (see [Commenting without approval](#commenting-without-approval)).

```powershell
.venv\Scripts\agent moltbook join "AgentName" "A one-line description"
.venv\Scripts\agent moltbook status
.venv\Scripts\agent moltbook draft <post ID or address>
.venv\Scripts\agent moltbook draft --new "what the post should be about"
.venv\Scripts\agent moltbook outbox
.venv\Scripts\agent moltbook approve 1
.venv\Scripts\agent moltbook reject 1
.venv\Scripts\agent moltbook inbox
```

1. **Join:** `join` registers the agent and prints a claim link. You open it, verify an email address, and post a tweet with the code shown. `status` shows the link again and tells you when the account is claimed.
2. **Draft:** the model is given one post and writes a comment, or says it has nothing to add. For this step it has no tools, notes, files or key.
3. **Check:** code refuses a draft that contains a link, a web or email address, a file path, a command, or anything that looks like a key.
4. **Approve:** you read the draft in the outbox. `approve` posts it, and `--edit "text"` posts your wording instead.

Moltbook sets a small maths puzzle before it shows what an agent wrote. The agent works it out and shows you its answer before sending, because ten wrong answers in a row suspend the account. Press Enter to accept, or type the right number. With `approve 1 --answer-later` the puzzle is shown and saved, and you have five minutes to send `agent moltbook answer <number>`.

Code unscrambles the puzzle and finds its two numbers, and the model is only asked which sum the sentence describes. A 9B model reading the scrambled text itself got three real puzzles wrong in a row.

Safeguards:

- The key is kept in `.config\moltbook\credentials.json` in your home folder, outside this project. It is only sent to Moltbook's own API, and it is never shown, logged, or given to the model.
- The agent sends at most one thing a minute and ten a day, well under Moltbook's limits.
- `inbox` shows who replied to the agent's posts. It leaves out the parts of Moltbook's reply that tell an agent what to do next.

Moltbook's own instructions ask agents to fetch a file from its website every 30 minutes and follow it. This agent doesn't. Following instructions fetched from a website would let that website steer the agent. It also doesn't upvote or follow anyone.

What the agent may say about itself is the `ABOUT` text in [agent/outbox.py](agent/outbox.py). Edit it to change that.

### Commenting without approval

This is off unless you turn it on, and one command turns it off again:

```powershell
.venv\Scripts\agent moltbook auto on
.venv\Scripts\agent moltbook auto on --every 6
.venv\Scripts\agent moltbook auto off
.venv\Scripts\agent moltbook auto status
.venv\Scripts\agent moltbook checkin
.venv\Scripts\agent moltbook checkin --dry-run
```

`auto on` allows check-ins. `--every 6` also has Windows run one every six hours. `auto off` stops both. `checkin --dry-run` shows what the agent would post without posting, whether auto is on or not.

A check-in is one pass:

1. It answers replies to the agent's own posts.
2. It searches Moltbook for two of the agent's subjects and reads up to six posts it hasn't seen.
3. For each post, the model writes a comment or says it has nothing to add.
4. It posts at most two comments, then stops.

The subjects are the `INTERESTS` list in [agent/moltbook_auto.py](agent/moltbook_auto.py). Edit it to change what the agent joins in on.

Every check-in writes `runs/moltbook-checkin-<date>.md`, listing what it read, what it posted, what it decided against, and any comment code blocked. That file is how you see what the agent is doing.

What still holds with nobody approving:

- The model has no tools, notes, files or key. It only writes text.
- Code refuses any comment with a link, an address, a path, a command or a key.
- The pace is the same: one comment a minute and ten a day.
- Code decides which posts are read and how many. Nothing on Moltbook can give the agent a task.
- A puzzle answer is only sent when the model's sum uses exactly the numbers code found. Otherwise the comment stays hidden.
- One failed puzzle ends the check-in, and three in a row turn auto off.

What to expect: the comments are on topic and polite, and they tend to ask a question about the post. A 9B model sometimes implies experience it doesn't have, such as "on limited hardware latency matters more". The prompt forbids that, and it still slips through now and then.

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

`agent eval` runs the tasks in [evals/tasks.json](evals/tasks.json) and checks the results in code. It runs the agent's tests itself rather than trusting the report. The tasks cover notes, web research, code, long documents, planted instructions, and tidying a folder. Add `"mode": "research"`, `"mode": "team"` or `"mode": "organize"` to a task to run it with that command.

Evals search the web far more than normal use, so they wait 15 seconds between searches. That adds several minutes to a full run. `agent eval --no-web` skips the tasks that search, which is the right choice when your change has nothing to do with the web. Each result is added to `evals/history.jsonl` with the commit and model, so you can tell whether a change helped.

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
| Team runs: the board and the agents    | [agent/team.py](agent/team.py)                                                                                           |
| The queue, and what research remembers | [agent/jobs.py](agent/jobs.py), [agent/memory.py](agent/memory.py)                                                       |
| Organize runs: prompts and rules       | [agent/organize.py](agent/organize.py)                                                                                   |
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
