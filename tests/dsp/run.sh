#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"${CC:-cc}" -std=c11 -O1 -g -Wall -Wextra -Werror -ffp-contract=off -no-pie \
  -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer \
  -I "$root/src/tiny_voca" "$root/tests/dsp/checked.c" "$root/src/tiny_voca/vocoder_dsp.c" \
  -lm -pthread -o "$work/checked"
"$work/checked"
