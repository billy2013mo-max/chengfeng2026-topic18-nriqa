import zipfile, os, sys, time
zpath, outdir = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(zpath)
names = [n for n in z.namelist() if not n.startswith('__MACOSX') and not n.endswith('/')]
t0 = time.time()
for i, n in enumerate(names):
    dst = os.path.join(outdir, os.path.basename(n))
    if os.path.exists(dst) and os.path.getsize(dst) == z.getinfo(n).file_size:
        continue
    with z.open(n) as f, open(dst, 'wb') as g:
        g.write(f.read())
    if (i+1) % 300 == 0:
        print(i+1, 'done', '%.1fs' % (time.time()-t0), flush=True)
print('total', len(names), '%.1fs' % (time.time()-t0))
