import pickle
import numpy as np
import random
import settings as ss

with open(ss.data_path + "dla_total_" + str(ss.la_total) + "_dist_" + str(ss.la_remove_dist_th) + "_ang_" + str(ss.la_remove_ang_th), "rb") as f:
    la = pickle.load(f)

la.sort(key=lambda l:l[1])
la_group = []
i = 0
while i < len(la):
    la_item = []
    tag = la[i][1]
    while tag == la[i][1]:
        la_item.append(la[i])
        i += 1;
        if i == len(la):
            break
    la_group.append(la_item)

rand_group = random.sample(la_group, ss.la_per_sample)
rand_la = []
for i in range(0, len(rand_group)):
    for j in range(0, len(rand_group[i])):
        rand_la.append(rand_group[i][j])

with open(ss.data_path + "rdla_total_" + str(ss.la_total) + "_sample_" + str(ss.la_per_sample), "wb") as f:
    pickle.dump(rand_la, f)
