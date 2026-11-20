# Run notes
- Host: Intel Xeon @ 2.10GHz VM, 4 cores (see host.txt).
- Tooling: git/cmake/g++/curl/python3 were present. Installed `gawk` via apt-get: the default awk (mawk) lacks `asort`, which run.sh's timing step needs (first attempt aborted there). run.sh itself was NOT modified.
- First attempt used `run.sh .` (OUT_DIR=. wrote files into repo root); those were deleted and the run repeated with default OUT_DIR. Results here are from the second run.
- Patches applied and builds succeeded without edits.
