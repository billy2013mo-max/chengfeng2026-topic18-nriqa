"""Create remote GitHub repository via `gh` CLI and push the local repository (`setup_git_repo.py`)."""
import glob
import os
import subprocess

os.chdir('/wanqing-develop/luowenjing/18')


def run(cmd):
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def main():
    print('=== 1. Checking sibling repo remotes ===')
    for d in ['Qwen3.8-27B-optimization', 'Param_Recommend', 'lmj', 'zpy']:
        cfg = f'/wanqing-develop/luowenjing/{d}/.git/config'
        if os.path.exists(cfg):
            _, out, _ = run(f'git -C /wanqing-develop/luowenjing/{d} remote -v')
            print(f'  {d}: {out}')

    print('\n=== 2. Checking existing GitHub repos for billy2013mo-max ===')
    _, gh_repos, gh_err = run('gh repo list --limit 10')
    print(gh_repos or gh_err)

    repo_name = 'chengfeng2026-topic18-nriqa'
    print(f'\n=== 3. Creating private GitHub repo {repo_name} and pushing ===')
    run('gh auth setup-git')
    rc, out, err = run(
        f'gh repo create {repo_name} --private --source=. --remote=origin --push '
        f'--description "2026乘风挑战赛-命题18：晴羽观感体系·画质BadCase的无参检测模型 (V9 High-Res Spatial-Anomaly Specialists)"'
    )
    print('gh repo create rc:', rc)
    print('stdout:', out)
    print('stderr:', err)
    if rc != 0:
        # If repo already exists, set remote and push
        run(f'git remote add origin https://github.com/billy2013mo-max/{repo_name}.git || true')
        rc2, out2, err2 = run('git push -u origin main')
        print('git push rc:', rc2, out2, err2)

    _, rem, _ = run('git remote -v')
    print('\nGit remote:', rem)

    print('\n=== 4. Current 8-GPU V9 Full-Data Training Progress ===')
    for log_p in sorted(glob.glob('results/v9_fulldata/*.log')):
        lines = open(log_p).read().splitlines()
        ep_lines = [l for l in lines if 'ep ' in l or 'DONE' in l or 'saved' in l or 'trainable' in l]
        print(f'  {os.path.basename(log_p):32s}: {ep_lines[-1] if ep_lines else (lines[-1] if lines else "starting")}')


if __name__ == '__main__':
    main()
