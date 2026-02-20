#!/usr/bin/env bash
# chmod +x codex-runner.sHello. h
# ./codex-runner.sh tasks/fixbug.md ~/projects/myrepo logs/fixbug.log
# alias run-task='~/bin/codex-runner.sh'
# run-task tasks/task1.md ~/projects/repo logs/task1.log




set -e

QUERY_MD="$1"
WORKSPACE="$2"
LOGFILE="$3"

if [ -z "$QUERY_MD" ] || [ -z "$WORKSPACE" ] || [ -z "$LOGFILE" ]; then
    echo "Usage: codex-runner <task.md> <workspace> <logfile>"
    exit 1
fi

if [ ! -f "$QUERY_MD" ]; then
    echo "Task file not found: $QUERY_MD"
    exit 1
fi

if [ ! -d "$WORKSPACE" ]; then
    echo "Workspace not found: $WORKSPACE"
    exit 1
fi

mkdir -p "$(dirname "$LOGFILE")"

cd "$WORKSPACE"

echo "== Codex Runner =="
echo "Workspace: $(pwd)"
echo "Task: $QUERY_MD"
echo "Log: $LOGFILE"
echo "==================="

PROMPT="$(cat "$QUERY_MD")"
codex exec --full-auto -s workspace-write -C "$WORKSPACE" "$PROMPT" 2>&1 | tee "$LOGFILE"

