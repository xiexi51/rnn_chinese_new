import pickle
import numpy as np
import settings as ss

with open(ss.data_path + "rdla_total_" + str(ss.la_total) + "_sample_" + str(ss.la_per_sample), "rb") as f:
    la = pickle.load(f)

y = []
x = []
j = 0
i = 0
while i < len(la):
    # tagbytes = la[i][1].encode("gb18030")
    # tag = tagbytes[0] * 256 + tagbytes[1]
    # taglist.append(tag)
    curtag = la[i][1]
    cur_y = np.zeros(ss.la_total)
    cur_y[j] = 1
    while la[i][1] == curtag:
        x.append(la[i][2])
        y.append(cur_y)
        i += 1
        if i == len(la):
            break
    j += 1

z = sorted(zip(x, y), key=lambda l:np.size(l[0], 0))
result = zip(*z)

x, y = [list(i) for i in result]

with open(ss.data_path + "x_y_la_n_" + str(ss.la_total) + "_s_" + str(ss.la_per_sample) + "_dist_" + str(ss.la_remove_dist_th) + "_ang_" + str(ss.la_remove_ang_th), "wb") as f:
    pickle.dump((x, y), f)

#_, _taglist = np.unique(taglist, return_inverse=True)

