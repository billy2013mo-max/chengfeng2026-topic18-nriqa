cd /wanqing-develop/luowenjing/18
while pgrep -f "recipe ogk --n" >/dev/null; do sleep 30; done
export PYTHONPATH=/wanqing-develop/luowenjing/18/code
python3 code/r55/build_gen_inpaint.py --recipe ogk --merge --nshards 1
python3 code/r55/prep_aux.py --gpus 1 --gpu-list 0 --steps template meta --manifest data/r55_aux/manifest_ogk.csv --suffix _ogk --tag r55ogk
python3 code/r55/synth_fidelity.py --recipe ogk --gpu 0
