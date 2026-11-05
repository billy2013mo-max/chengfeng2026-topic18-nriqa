"""Initialize local Git repository, inspect available git/gh credentials or remotes,
create a clean .gitignore (excluding multi-GB weights/images while tracking all code,
JSON results, and submission files), and commit the repository (`setup_git_repo.py`).
"""
import os
import subprocess

os.chdir('/wanqing-develop/luowenjing/18')


def run(cmd):
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def main():
    print('=== 1. Checking GPU status ===')
    _, out, _ = run('nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader')
    print(out)

    print('\n=== 2. Checking existing git/gh/ssh configs ===')
    for path in ['/root/.gitconfig', '/root/.git-credentials', '/root/.ssh/config', '/root/.config/gh/hosts.yml']:
        print(f'  exists({path}) = {os.path.exists(path)}')
    _, gh_out, gh_err = run('gh auth status')
    print('  gh auth status:', gh_out or gh_err)
    _, ssh_ls, _ = run('ls -la /root/.ssh')
    print('  /root/.ssh:\n', ssh_ls)
    _, parent_git, _ = run('find /wanqing-develop/luowenjing -maxdepth 2 -name ".git"')
    print('  parent .git dirs:', parent_git)

    print('\n=== 3. Creating .gitignore ===')
    gitignore = """# Large pretrained weights & checkpoints
models/
*.pt
*.pth
*.bin
*.safetensors
*.onnx

# Raw image datasets & synthetic images
data/raw/
data/images/
data/synth*/
data/v9_fulldata/synth*/
**/synth_*/
*.jpg
*.jpeg
*.png
*.webp
*.bmp
*.zip
*.tar
*.tar.gz
*.tgz

# Large numpy probability/feature caches (keep lightweight .json summaries & submissions)
*.npz
*.npy
data/*.csv

# Logs & temporary files
__pycache__/
*.pyc
*.log
.DS_Store
tmp/
"""
    with open('.gitignore', 'w') as f:
        f.write(gitignore)

    print('\n=== 4. Initializing Git repository & committing code + submissions ===')
    if not os.path.exists('.git'):
        run('git init -b main || git init')
    _, name, _ = run('git config user.name')
    if not name:
        run('git config user.name "luowenjing"')
        run('git config user.email "luowenjing@wanqing-develop.local"')

    run('git add .gitignore code/ pred_nriqa_dataset_test*.json results/*/*.json *.md')
    _, status_out, _ = run('git status --short')
    print(f'Staged files count: {len(status_out.splitlines())}')
    rc, commit_out, commit_err = run('git commit -m "feat: V9 High-Res Spatial-Anomaly Specialists & 8-Fold Full-Data NR-IQA Pipeline"')
    print('git commit:', commit_out or commit_err)
    _, log_out, _ = run('git log -n 2 --stat')
    print('\nLatest commit summary:\n', '\n'.join(log_out.splitlines()[:25]))


if __name__ == '__main__':
    main()
