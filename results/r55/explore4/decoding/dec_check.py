# Check fast scorer == common55.score; time it; baseline nested decoders.
import time, numpy as np
import dec_lib as L
D = L.load(); g = D["gold"]; P = D["P"]
rs = np.random.RandomState(0)
for _ in range(20):
    pr = np.where(rs.rand(*g.shape) < 0.3, rs.randint(0, 4, g.shape), g); pr[:, [3, 5]] = np.where(pr[:, [3, 5]] > 0, 3, 0)
    a, b = L.fscore(pr, g), L.c.score(pr, g)[L.K]
    assert abs(a - b) < 1e-9, (a, b)
t = time.time(); [L.fscore(pr, g) for _ in range(200)]; print("fast score ms", (time.time() - t) / 200 * 1e3)
am = L.d55.argmax_decode(P)
print("argmax", L.fscore(am, g), L.fscore(am[D["orig"]], g[D["orig"]]))
t = time.time()
single = L.nested(lambda a, b, x, i, j: L.d55.Decoder55(L.K).fit(a, b).predict(x), D)
print("nested single", L.fscore(single, g), L.fscore(single[D["orig"]], g[D["orig"]]), time.time() - t, "s")
t = time.time()
ms = L.nested(lambda a, b, x, i, j: L.d55_multistart(a, b).predict(x), D)
print("nested multistart", L.fscore(ms, g), L.fscore(ms[D["orig"]], g[D["orig"]]), time.time() - t, "s")
ins = L.d55_multistart(P, g).predict(P); print("insample multistart", L.fscore(ins, g), L.fscore(ins[D["orig"]], g[D["orig"]]))
np.savez(L.HERE + "/dec_base_preds.npz", argmax=am, single=single, multistart=ms, insample=ins)
