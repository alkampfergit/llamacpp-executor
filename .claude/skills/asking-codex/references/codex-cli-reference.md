# Codex CLI reference

Verified against **`codex-cli 0.149.0`** on Windows (npm global install). Flag availability
changes between versions — re-check with `--help` if something is rejected.

## Contents

- [Subcommands](#subcommands)
- [`codex exec` flags](#codex-exec-flags)
- [`codex exec review` flags — and what it refuses](#codex-exec-review-flags--and-what-it-refuses)
- [Model catalog](#model-catalog)
- [Reasoning effort](#reasoning-effort)
- [Config and auth locations](#config-and-auth-locations)
- [Verified failure signatures](#verified-failure-signatures)

---

## Subcommands

`codex <cmd>`; run with no subcommand for the interactive TUI.

| Command | Purpose |
| --- | --- |
| **`exec`** (alias `e`) | Run non-interactively. **This is the one to automate.** |
| **`review`** / `exec review` | Code review against the repository. |
| `debug models` | **Render the raw model catalog as JSON.** There is no `codex models`. |
| `debug prompt-input` | Render the model-visible prompt input as JSON. |
| `doctor` | Diagnose install, config, auth, runtime health. Prints the real `codex.exe` path. |
| `login` / `logout` | Manage authentication. |
| `mcp` / `mcp-server` | Manage external MCP servers / run Codex itself as an MCP server. |
| `apply` (alias `a`) | `git apply` the latest diff Codex produced. |
| `resume` / `fork` | Continue or branch a previous session (`--last` for the most recent). |
| `sandbox` | Run arbitrary commands inside Codex's sandbox. |
| `update` | Self-update. |
| `plugin`, `features`, `agents`, `cloud`, `app` | Plugins, feature flags, session browser, Codex Cloud, desktop app. |

---

## `codex exec` flags

| Flag | Notes |
| --- | --- |
| `[PROMPT]` | Positional. Use `-` (or omit) to read from **stdin**. If stdin is piped *and* a prompt is given, stdin is appended as a `<stdin>` block. |
| `-o, --output-last-message <FILE>` | **Use this for results.** The CLI writes the agent's final message here. Works under `read-only`. |
| `-s, --sandbox <MODE>` | `read-only` \| `workspace-write` \| `danger-full-access` |
| `-m, --model <MODEL>` | Per-invocation model. |
| `-c, --config <key=value>` | Override any config key, dotted path, TOML-parsed value. E.g. `-c model_reasoning_effort=high`. |
| `-p, --profile <NAME>` | Layer `$CODEX_HOME/<name>.config.toml` over the base config. |
| `--ephemeral` | Don't persist session files. Good for one-shot automation. |
| `--json` | Emit events to stdout as JSONL. |
| `--output-schema <FILE>` | JSON Schema constraining the final response shape. |
| `-C, --cd <DIR>` | Working root. Default is the current directory. |
| `--add-dir <DIR>` | Extra writable directory. |
| `--skip-git-repo-check` | Permit running outside a git repo. |
| `--ignore-user-config` | Skip `$CODEX_HOME/config.toml` (auth still uses `CODEX_HOME`). |
| `--ignore-rules` | Skip user/project execpolicy `.rules`. |
| `-i, --image <FILE>...` | Attach images. |
| `--oss` / `--local-provider <lmstudio\|ollama>` | Use a local provider. |
| `--enable` / `--disable <FEATURE>` | Feature flags (`-c features.<name>=…`). |
| `--strict-config` | Error on unrecognised config keys. |
| `--color <always\|never\|auto>` | |
| `--approve-for-me` | Route approvals through automatic review under `workspace-write`. |
| ⚠️ `--dangerously-bypass-approvals-and-sandbox` | No sandbox, no prompts. Only with explicit user authorisation. |
| ⚠️ `--dangerously-bypass-hook-trust` | Runs hooks without trust verification. |

---

## `codex exec review` flags — and what it refuses

Scope selection:

| Flag | Reviews |
| --- | --- |
| `--uncommitted` | staged + unstaged + untracked |
| `--base <BRANCH>` | current branch against a base |
| `--commit <SHA>` | one commit |
| `[PROMPT]` | custom review instructions; Codex chooses scope |

Also accepted: `-m`, `-o`, `-c`, `--json`, `--ephemeral`, `--output-schema`, `--title`,
`--enable`/`--disable`, `--skip-git-repo-check`, `--ignore-user-config`, `--ignore-rules`,
`--strict-config`, and the two dangerous bypasses.

**Two restrictions, both found by testing rather than documentation:**

1. **No `-s/--sandbox`** (nor `-C/--cd`, nor `--add-dir`). Review is read-only by nature:
   ```
   error: unexpected argument '-s' found
   ```
2. **A scope flag cannot be combined with `[PROMPT]`.** Verified for all three:
   ```
   error: the argument '--uncommitted' cannot be used with '[PROMPT]'
   error: the argument '--base <BRANCH>' cannot be used with '[PROMPT]'
   error: the argument '--commit <SHA>' cannot be used with '[PROMPT]'
   ```
   Choose scope **or** instructions. If you need both, use plain `codex exec` — it has no such
   restriction and can read the repo itself under `read-only`.

---

## Model catalog

From `codex debug models`. **All nine have a 272,000-token context window.**

| slug | display | visibility | description |
| --- | --- | --- | --- |
| `gpt-5.6-sol` | GPT-5.6-Sol | list | Latest frontier agentic coding model |
| `gpt-5.6-terra` | GPT-5.6-Terra | list | Balanced agentic coding for everyday work |
| `gpt-5.6-luna` | GPT-5.6-Luna | list | Fast and affordable agentic coding |
| `gpt-daybreak-blue-latest` | Daybreak Blue | list | Frontier model for broad **defensive cybersecurity** work |
| `gpt-5.5` | GPT-5.5 | list | Frontier model for complex coding, research, real-world work |
| `gpt-5.4` | GPT-5.4 | list | Strong model for everyday coding |
| `gpt-5.4-mini` | GPT-5.4-Mini | list | Small, fast, cost-efficient for simpler tasks |
| `gpt-reserve` | GPT-Reserve | **hide** | Fast and affordable |
| `codex-auto-review` | Codex Auto Review | **hide** | Automatic approval-review model |

`hide` means absent from the interactive picker; all nine report `supported_in_api: true` and
can be passed explicitly with `-m`.

Other fields per model worth knowing: `default_reasoning_level`, `supported_reasoning_levels`,
`max_context_window`, `input_modalities`, `supports_search_tool`, `apply_patch_tool_type`,
`priority` (picker ordering — `gpt-5.6-sol` is 1).

---

## Reasoning effort

Set with `-c model_reasoning_effort=<level>`.

| Family | Levels |
| --- | --- |
| `gpt-5.6-*`, `gpt-daybreak-blue-latest` | `low, medium, high, xhigh, max, **ultra**` |
| `gpt-reserve`, `codex-auto-review` | `low, medium, high, xhigh, max` |
| `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini` | `low, medium, high, xhigh` |

Note `ultra` exists here and has no equivalent in Claude Code's effort scale. Each level
carries its own description, e.g. `low` → *"Fast responses with lighter reasoning"*.

---

## Config and auth locations

| Path | Contents |
| --- | --- |
| `$CODEX_HOME` (default `~\.codex`) | Everything below |
| `~\.codex\config.toml` | `model`, `model_reasoning_effort`, `[mcp_servers.*]`, `[projects.*]`, `[plugins.*]` |
| `~\.codex\auth.json` | Stored credentials (ChatGPT or API login) |
| `~\.codex\<name>.config.toml` | Profile overlays for `-p/--profile` |
| `~\.codex\skills\` | Global Codex-side skills |
| `<repo>\.agents\skills\`, `<repo>\.codex\skills\` | Repo-local Codex-side skills |

⚠️ `config.toml` can hold plaintext API keys for MCP servers. Treat it as a secret, keep it out
of screenshots and backups, and prefer environment variables where the server supports them.

Codex-side skills are activated **by prompt text** (`Use $skill-name …`), not by a flag, so a
skill reference must be part of the prompt you send.

---

## Verified failure signatures

| Symptom | Cause |
| --- | --- |
| `The specified executable is not a valid application for this OS platform` | You launched the npm **shim** (`codex.ps1` / `codex.cmd`) via a direct process start. Resolve the real `codex.exe` — `codex doctor` prints its path. |
| `Error: stdin is not a terminal` | Interactive mode with no TTY. Use `codex exec`, and pass `-` with piped stdin. |
| `error: unexpected argument '-s' found` | `-s` given to `exec review`, which has no sandbox flag. |
| `error: the argument '--base <BRANCH>' cannot be used with '[PROMPT]'` | Scope flag combined with custom instructions in review mode. |
| Empty or absent `-o` file, exit 0 | The agent produced no final message. Report stdout/stderr; never invent a result. |
| Run killed at ~120 s | Tool-level timeout, not Codex. `exec` is synchronous; raise the timeout (a trivial prompt is ~6–10 s, a repo review 30 s to many minutes). |
