cd /wanqing-develop/luowenjing/18
while pgrep -f "recipe ogkup --n" >/dev/null; do sleep 30; done
export PYTHONPATH=/wanqing-develop/luowenjing/18/code
python3 code/r55/build_gen_inpaint.py --recipe ogkup --merge --nshards 1
python3 code/r55/prep_aux.py --gpus 1 --gpu-list 1 --steps template meta --manifest data/r55_aux/manifest_ogkup.csv --suffix _ogkup --tag r55ogkup
python3 code/r55/synth_fidelity.py --recipe ogkup --gpu 1
