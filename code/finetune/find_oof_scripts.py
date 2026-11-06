import glob
for p in sorted(glob.glob('code/**/*.py', recursive=True)):
    with open(p, 'r', errors='ignore') as f:
        txt = f.read()
    if 'internvit-6b_combined' in txt or '_combined_oof' in txt or 'hand_features' in txt:
        print("MATCH:", p)
