#!/usr/bin/env bash
# queue.sh TAG ITERS pairs...: waits until no other batch runs, then runs batch.sh (one pinned run at a time).
W=$(cd "$(dirname "$0")/.." && pwd)
while pgrep -f "tools/batch.sh" > /dev/null; do sleep 3; done
exec $W/tools/batch.sh "$@"
