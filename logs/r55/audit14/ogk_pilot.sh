cd /wanqing-develop/luowenjing/18
for s in 0 1; do CUDA_VISIBLE_DEVICES=$s venv_flux/bin/python -I code/r55/build_gen_inpaint.py --recipe ogk --n 48 --shard $s --nshards 2 > logs/r55/audit14/gen_ogk_s$s.log 2>&1 & done; wait
export PYTHONPATH=/wanqing-develop/luowenjing/18/code
python3 code/r55/build_gen_inpaint.py --recipe ogk --merge --nshards 2
python3 code/r55/prep_aux.py --gpus 1 --gpu-list 0 --steps template meta --manifest data/r55_aux/manifest_ogk.csv --suffix _ogk --tag r55ogk
python3 code/r55/synth_fidelity.py --recipe ogk --gpu 0
