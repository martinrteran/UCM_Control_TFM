---
name: project-briefing
description: Load or refresh PROJECT_BRIEFING.md, the file with the repo's current state (in-flight refactors, broken scripts, experiment history, next steps). Use at the start of a session (load) or after changing code or finishing an experiment (update).
argument-hint: "[load|update]"
---

# Project Briefing

Fast context-load and context-refresh for `PROJECT_BRIEFING.md`, the file
that lets an agent get up to speed on this repo's *current* state (in-flight
refactors, broken scripts, experiment history, next steps) without
re-reading the whole codebase.

`$ARGUMENTS`:
- empty / `load` / `catch-up` → **load mode** (default)
- `update` / `refresh` → **update mode**

## Load mode (default)

1. Read `PROJECT_BRIEFING.md` at the repo root. If it doesn't exist, tell the
   user and offer to generate one (see Update mode, step by step, run on the
   whole repo instead of just the diff).
2. Read `CLAUDE.md` if it isn't already in context — `PROJECT_BRIEFING.md`
   assumes it and doesn't repeat static architecture info.
3. Give the user a short (5–8 line) spoken summary: current branch/commit,
   whether the working tree has uncommitted work that changes behavior (§2 of
   the briefing), the single most important "next step" line, and anything
   flagged as currently broken.
4. Do **not** re-derive this from scratch by grepping the codebase — the
   point of this command is to skip that. Only fall back to exploring the
   code directly if the briefing is silent on something you need, or if
   something it claims looks stale (see step 2 of Update mode for how to
   tell).

## Update mode

Run this after any session that changed code, added/finished an experiment,
or uncovered something about the project worth remembering for next time.

1. **Read the current `PROJECT_BRIEFING.md`** in full, and note its
   "Last updated" / "Branch" / "Latest commit" line.

2. **Figure out what changed since then**:
   - `git log --oneline <last-known-commit>..HEAD` (or `git log -15` if the
     briefing predates the current branch) for new commits.
   - `git status --short` for uncommitted changes, filtering out noise paths
     (`runs/`, `checkpoints/`, `__pycache__/`, `*.pyc`) — those are training
     artifacts, not code changes worth narrating.
   - `git diff --stat HEAD` for the shape of uncommitted work.
   - If the session added a new `train/grid_training_NN.py`, check whether
     `runs/Test/NN/` or `checkpoints/Test/NN/` now exist — that's how you
     tell whether an experiment actually ran vs. was only written.

3. **Sanity-check claims already in the briefing** that this session's work
   might have invalidated — e.g. if §2 says a script is "blocked" because an
   env requires `reward_fn`, check whether that's still true before carrying
   the claim forward verbatim. A stale "known issue" is worse than no issue
   listed.

4. **Edit `PROJECT_BRIEFING.md` in place** (don't rewrite sections that are
   still accurate):
   - Bump the "Last updated" / branch / commit line at the top.
   - §2 ("Current working-tree state"): update or remove resolved items, add
     newly discovered gotchas. If the mid-refactor situation got resolved
     (e.g. a `reward_fn` was finally written and wired in), replace that
     whole subsection with what actually happened and what config/behavior
     resulted — future agents need the *current* contract, not the history
     of getting there (that belongs in commit messages).
   - §3 (experiment history table): append a row for any new test number,
     using the actual commit message as the "commit note" column and
     checking `runs/Test/NN` + `checkpoints/Test/NN` for whether it has
     output yet.
   - §4/§5 (architecture/obs-action cheat sheet): only touch if the session
     changed a class signature, added/removed a trainer, changed the
     observation layout, etc. — these should rarely need edits.
   - Keep it a state snapshot, not a changelog: don't accumulate "on date X
     we also noticed Y" entries that never get cleaned up. Each section
     should read as true *right now*.

5. Tell the user what you updated in 2–3 lines. Don't commit the file
   yourself unless asked — let the user decide when to commit documentation
   changes alongside (or separately from) the code changes.
