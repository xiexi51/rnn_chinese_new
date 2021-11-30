import pickle
import numpy as np
import random
import settings as ss

with open(ss.data_path + "dlb_total_" + str(ss.lb_total) + "_dist_" + str(ss.lb_remove_dist_th) + "_ang_" + str(ss.lb_remove_ang_th), "rb") as f:
    lb = pickle.load(f)

random.seed(123)
lb2 = random.sample(list(filter(lambda r: r[0] == 1001, lb)), ss.nclass)
y = []
x = []
i = 0
for i in range(ss.nclass):
    char = lb2[i]
    for j in range(ss.repeat):
        k = 0
        first = True
        sum_p1 = np.zeros(2)
        for k in range(np.size(char[2], 0)):
            p1 = np.array(char[2][k])
            p2 = np.array(char[3][k])

            if random.random() < ss.drop:
                if p2[0] == 1:
                    p2[0] = 0
                    p2[1] = 1
            if random.random() < ss.noise_prob:
                p1[0] *= 1 + (random.random() - 0.5) * 2 * ss.noise_ratio
                p1[1] *= 1 + (random.random() - 0.5) * 2 * ss.noise_ratio

            sum_p1 = sum_p1 + p1
            if first:
                first = False
                cx = np.expand_dims([i * 1.0], 0)
                cy = [np.append(sum_p1, p2)]
            else:
                cx = np.append(cx, np.expand_dims([i * 1.0], 0), 0)
                cy = np.append(cy, [np.append(sum_p1, p2)], 0)
        x.append(cx)
        y.append(cy)

z = sorted(zip(x, y), key=lambda l:np.size(l[0], 0))
result = zip(*z)
x, y = [list(i) for i in result]
batchx = []
batchy = []
i = 0
while i < x.__len__():
    xb = np.expand_dims(x[i], 0)
    yb = np.expand_dims(y[i], 0)
    j = i + 1
    while j < x.__len__():
        if np.size(x[j], 0) == np.size(x[i], 0):
            xb = np.append(xb, np.expand_dims(x[j], 0), 0)
            yb = np.append(yb, np.expand_dims(y[j], 0), 0)
            j += 1
        else:
            break
    i = j
    batchx.append(xb)
    batchy.append(yb)

with open(ss.data_path + "x_y_lb100_n_" + str(ss.nclass) + "_r_" + str(ss.repeat) + "_dist_" + str(ss.lb_remove_dist_th) + "_ang_" + str(ss.lb_remove_ang_th) + "_drop_" + str(ss.drop) + "_np_" + str(ss.noise_prob) + "_nr_" + str(ss.noise_ratio), "wb") as f:
    pickle.dump((batchx, batchy), f)