# count.py ref.cpp cand.cpp... : changed lines vs the original file, code / comment-only / blank, as git diff counts them
import subprocess, sys
ref = sys.argv[1]
print(f"{'cand':10} {'+code':>6} {'-code':>6} {'+cmt':>5} {'-cmt':>5} {'+blank':>6} {'-blank':>6} {'total+/-':>9}")
for c in sys.argv[2:]:
    d = subprocess.run(["git", "diff", "--no-index", "-U0", ref, c], capture_output=True, text=True).stdout.splitlines()
    n = dict(pc=0, mc=0, pk=0, mk=0, pb=0, mb=0)
    for l in d:
        if l.startswith(("+++", "---")) or not l[:1] in "+-":
            continue
        body = l[1:].strip()
        kind = "b" if not body else ("k" if body.startswith("//") else "c")
        n[("p" if l[0] == "+" else "m") + kind] += 1
    name = c.split("/")[-1][:-4]
    print(f"{name:10} {n['pc']:6} {n['mc']:6} {n['pk']:5} {n['mk']:5} {n['pb']:6} {n['mb']:6} {'+%d/-%d' % (n['pc']+n['pk']+n['pb'], n['mc']+n['mk']+n['mb']):>9}")
