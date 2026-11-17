import sys,os
# enqueue.py PRIORITY NAME BACKBONE "EXTRA" [folds]
pri,name,bb,extra=sys.argv[1:5]; folds=sys.argv[5].split(",") if len(sys.argv)>5 else [str(i) for i in range(8)]
Q="/wanqing-develop/luowenjing/18/results/r55/sprint_os_og/queue/pending"
for f in folds:
    cmd=(f"python3 -u code/r55/train_testbed.py --backbone {bb} --fold {f} --gpu 0 --seed 2026 "
         f"--out-dir results/next55/towers/{name} --select fixed --fast-fixed --skip-t150 --no-save-ckpt --workers 6 {extra} "
         f"> results/r55/sprint_os_og/logs/{name}_f{f}.log 2>&1")
    open(f"{Q}/{pri}_{name}_f{f}","w").write(cmd)
print("enqueued",name,folds)
