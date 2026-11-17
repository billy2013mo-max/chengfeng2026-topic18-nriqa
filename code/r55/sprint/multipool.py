import os; os.environ["OMP_NUM_THREADS"]="2"
import sys, pickle, numpy as np
from harness import *
from dimpool import nested_dimpool
from ordinal_dec import nested_cum_bag
sys.path.insert(0,"/wanqing-develop/luowenjing/18/code/r55"); sys.path.insert(0,"/wanqing-develop/luowenjing/18/code"); sys.path.insert(0,"/wanqing-develop/luowenjing/18/code/next55")
from eval_specialist_gate import load_oof_dir
import common55 as c
T="/wanqing-develop/luowenjing/18/results/next55/towers/"
ext={"s26":"r55_sp_ntgpe_focus_s2026/pe_core_large","s27":"r55_sp_ntgpe_focus_s2027/pe_core_large","s28":"r55_sp_ntgpe_focus_s2028/pe_core_large","s29":"r55_sp_ntgpe_focus_s2029/pe_core_large","cnxf":"r55_sp_ntgcnx_focus_s2026/convnextv2_base"}
PX=dict(D["P"])
for k,v in ext.items(): a,ok=load_oof_dir(T+v,names); PX[k]=c.renorm(a)
res={}
for d in (2,4):
    pools=[D["Q"][:,d], nested_dimpool(D["P"],d)[0], nested_dimpool(PX,d)[0]]
    res[d]=nested_cum_bag(np.mean(pools,0),g[:,d])
report("multi-pool average (routedQ + dimpool v7 + dimpool v7+5) + bagged cum", res)
