import csv, json, sys
src, manifest, out = sys.argv[1], sys.argv[2], sys.argv[3]
keep = {r['name'] for r in csv.DictReader(open(manifest))}
n = 0
with open(out,'w') as f:
    for l in open(src):
        r = json.loads(l)
        if r['image'] in keep:
            f.write(l); n += 1
print(out, n)
