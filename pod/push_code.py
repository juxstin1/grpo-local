"""Send the Mission4 code, gym and SFT adapters from this PC to a pod (~95 MB, streamed tar over ssh).

  .venv\\Scripts\\python.exe pod\\push_code.py <ip> <ssh_port> [workdir]     (default /workspace/grpo)

Same file set as runs/audits/mission4-runpod-ready-2026-10-27/stage_payload.py, plus what that payload
missed on 2026-10-27 (gym data files, gym_budget.py/notes.py next to the gym) and the pod launcher.
Refuses to send if a qualified source drifted from qualification-v2.json.
"""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[1]
GYM_ROOT = REPO.parent / 'grpo-local-wt-filetools'
KEY = str(Path.home() / '.ssh/id_ed25519_runpod')
QUAL = 'runs/audits/mission4-tag-cleanup-2026-10-27/runtime/qualification-v2.json'
EXTRA = ['prompts/work-contract-v1.md', 'suite-v2/tasks.json', 'suite-training/v1/tasks.json',
         'suite-training/v1/registry.json', 'gym_budget.py', QUAL,
         'runs/audits/mission4-tag-cleanup-2026-10-27/runtime/admission.json',
         'runs/audits/mission4-run-2026-10-27/cache-stage.json',
         'runs/audits/mission4-pod-2026-10-27/mission4_pod.py',
         'runs/pilots/q4-bf16-gap-1010/lora-sft-1008.f16.gguf']
ADAPTER = 'adapters-sft-1008/lora-Qwen3.5-4B-sft-hosted-distil-1007'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    q = json.loads((REPO / QUAL).read_text())
    for rel, digest in q['source_sha256'].items():
        if sha(REPO / rel) != digest:
            raise SystemExit('qualified source drifted: ' + rel)
    out = {'repo/' + rel: REPO / rel for rel in q['source_sha256']}
    out.update({f'repo/ops/{p.name}': p for p in (REPO / 'ops').glob('*.py')})
    out.update({'repo/' + rel: REPO / rel for rel in EXTRA})
    out.update({f'repo/{ADAPTER}/{p.name}': p for p in (REPO / ADAPTER).iterdir() if p.is_file()})
    # Not-ignored files only (~183, 1.5 MB); the gym folder also holds ~11 GB of ignored run data.
    gym = subprocess.run(['git', '-C', str(GYM_ROOT), 'ls-files', '--cached', '--others', '--exclude-standard',
                          'sandbox-gym'], capture_output=True, text=True, check=True).stdout.split('\n')
    out.update({rel: GYM_ROOT / rel for rel in gym if rel and (GYM_ROOT / rel).is_file()})
    out.update({name: GYM_ROOT / name for name in ('gym_budget.py', 'notes.py')})
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    ip, port = sys.argv[1], sys.argv[2]
    work = sys.argv[3] if len(sys.argv) > 3 else '/workspace/grpo'
    if not work.startswith('/workspace/'):         # Git Bash rewrites /workspace/... to C:/Program Files/Git/...
        raise SystemExit(f'workdir must start with /workspace/ (got {work!r}); run from PowerShell')
    entries = files()
    size = sum(p.stat().st_size for p in entries.values())
    if size > 300e6:
        raise SystemExit(f'refusing to push {size / 1e6:.0f} MB; expected ~95 MB')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tar:
        for arc, path in sorted(entries.items()):
            tar.add(path, arcname=arc, recursive=False)
    manifest = ''.join(f'{sha(p)}  {arc}\n' for arc, p in sorted(entries.items()))
    remote = (f'mkdir -p {work} && cd {work} && tar --no-same-owner -xf - && '
              f'cat > results-push.sha256 <<"EOF"\n{manifest}EOF\n'
              f'sha256sum -c --quiet results-push.sha256 && echo "pushed {len(entries)} files, all sha256 OK"')
    ssh = ['ssh', '-i', KEY, '-p', port, '-o', 'StrictHostKeyChecking=accept-new', f'root@{ip}', remote]
    subprocess.run(ssh, input=buf.getvalue(), check=True)


if __name__ == '__main__':
    main()
