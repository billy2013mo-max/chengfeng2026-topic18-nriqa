import os, glob
os.chdir('/wanqing-develop/luowenjing/18')
for d in ['/tmp/synth/flux3', '/tmp/synth/out', '/tmp/v8_ladder_imgs', 'data/synth_overgen_pro_s16', 'data/synth_overgen_pro_s17', 'data/synth_overgen_pro_full', 'data/synth_v6_s16', 'data/synth_v6_s17', 'data/synth_v6_full']:
    if os.path.exists(d):
        fs = [f for f in os.listdir(d) if not f.endswith('.txt')]
        print(f"{d:32s}: {len(fs)} files (sample: {fs[:3]})")
    else:
        print(f"{d:32s}: MISSING")
