import numpy as np, pandas as pd, tempfile, os
from naturenet.models.irl.run_hiavi_multiagent import resolve_windows, resolve_window, draw_shifts, _to_days
from naturenet.models.irl import hiavi_multiagent as hm
d = list(pd.date_range("2015-07-07","2015-12-31")) + list(pd.date_range("2017-07-10","2017-12-11"))
f = tempfile.mktemp(suffix=".csv"); pd.DataFrame({"t":range(len(d)),"date":[x.date() for x in d],"complete":True}).to_csv(f,index=False)
ws = resolve_windows({"scene_index_csv": f, "split_gap_days": 30}); print(ws)
assert ws == [("2015-07-07","2015-12-31"),("2017-07-10","2017-12-11")]
print(resolve_window({"scene_index_csv": f}), resolve_windows(["2017-07-10","2017-12-11"]))
win = [(float(_to_days([a])[0]), float(_to_days([b])[0]) + 1.0) for a, b in ws]
sh = draw_shifts(["a"], np.random.default_rng(0), win, 20); print(sh)
class P: pass
p = P(); t = np.array([[win[0][0]+1, win[0][1]-1, win[1][0]+3, np.nan]]); p.times=t; p.animal=np.array(["a"])
o = hm.PartnerContext.shifted_times(p, sh, win); print(t, o)
for (a,b),x in zip([win[0],win[0],win[1]], o[0][:3]): assert a <= x < b
# single window back-compat
o1 = hm.PartnerContext.shifted_times(p, sh, win[1]); print(o1)
# several sequences / animals: (N,) animal ids against (N, T) step times
p2 = P(); p2.times = np.array([[win[0][0]+1, win[0][0]+2], [win[1][0]+1, win[1][0]+2], [win[0][0]+5, win[0][0]+6]])
p2.animal = np.array(["a", "b", "a"])
o2 = hm.PartnerContext.shifted_times(p2, {"a": 40.0, "b": 0.0}, win)
assert np.allclose(o2[1], p2.times[1]) and np.allclose(o2[[0, 2]], p2.times[[0, 2]] + 40.0), o2
print("OK")
