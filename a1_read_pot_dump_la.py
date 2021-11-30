import struct
import numpy as np
import matplotlib.pyplot as plt
import pickle
import settings as ss
import os

def remove_point_dist(_x, _y, _dist):
    i = 1
    while i < len(_x):
        if np.sqrt((_x[i] - _x[i - 1]) ** 2 + (_y[i] - _y[i - 1]) ** 2) < _dist:
            _x = np.delete(_x, [i])
            _y = np.delete(_y, [i])
            continue
        i += 1
    return _x, _y

def remove_point_ang(_x, _y, _tang):
    i = 1
    while i < len(_x) - 1:
        if ((_x[i]-_x[i-1])*(_x[i+1]-_x[i])+(_y[i]-_y[i-1])*(_y[i+1]-_y[i]))/np.sqrt(((_x[i]-_x[i-1])**2+(_y[i]-_y[i-1])**2)*((_x[i+1]-_x[i])**2+(_y[i+1]-_y[i])**2)) > _tang:
            _x = np.delete(_x, [i])
            _y = np.delete(_y, [i])
            continue
        i += 1
    return _x, _y

LA = []
for file in range(1001, 1001 + ss.la_total):
    filename = "e:\\tf_projects\\Pot1.1Train\\" + str(file) + ".pot"
    print(filename)
    show = 999999
    fbegin = 171
    fn = 99999
    fcnt = 0
    with open(filename, "rb") as f:
        for _ in range(0, fbegin):
            f.read(struct.unpack("h", f.read(2))[0] - 2)
        total = 0
        while total < fn:
            A = []
            A.append(file)
            _sample_size = f.read(2)
            if len(_sample_size) == 0:
                break
            sample_size = struct.unpack("h", _sample_size)[0]
            _tag_code = f.read(2)
            ba = bytearray(_tag_code)
            b2 = ba[1]
            ba[1] = ba[0]
            ba[0] = b2
            f.read(2)
            try:
                tag_code = ba.decode("gb18030")
            except Exception:
                print(str(total) + " gb2312 decode exception")
            else:
                print(str(total) + " " + tag_code)
            A.append(tag_code)
            stroke_number = struct.unpack("h", f.read(2))[0]
            strokex = []
            strokey = []
            origin_x = []
            origin_y = []
            for _ in range(0, stroke_number):
                sx = []
                sy = []
                while True:
                    px = struct.unpack("h", f.read(2))[0]
                    py = struct.unpack("h", f.read(2))[0]
                    if px == -1 and py == 0:
                        break
                    sx.append(px)
                    sy.append(py)
                _sx = np.array(sx, dtype=float)
                _sy = np.array(sy, dtype=float)
                origin_x.append(_sx)
                origin_y.append(_sy)
                if len(_sx) > 2:
                    _sx, _sy = remove_point_dist(_sx, _sy, ss.la_remove_dist_th)
                    if len(_sx) > 2:
                        _sx, _sy = remove_point_ang(_sx, _sy, ss.la_remove_ang_th)
                strokex.append(_sx)
                strokey.append(_sy)
            if total == show:
                fig1 = plt.figure()
                fig_origin = fig1.add_subplot(1,3,1)
                for i in range(0, len(origin_x)):
                    fig_origin.plot(origin_x[i], origin_y[i], color="black")

                fig_before_preprocess_plt = fig1.add_subplot(1, 3, 2)

                for i in range(0, len(strokex)):
                    fig_before_preprocess_plt.plot(strokex[i], strokey[i], color="black")
            lenl = []
            pxl = []
            pyl = []
            for i in range(0, len(strokex)):
                for j in range(0, len(strokex[i]) - 1):
                    _lenl = np.sqrt((strokex[i][j + 1] - strokex[i][j]) ** 2 + (strokey[i][j + 1] - strokey[i][j]) ** 2)
                    lenl.append(_lenl)
                    pxl.append(_lenl * (strokex[i][j + 1] + strokex[i][j]) / 2)
                    pyl.append(_lenl * (strokey[i][j + 1] + strokey[i][j]) / 2)
            ux = np.sum(pxl) / np.sum(lenl)
            uy = np.sum(pyl) / np.sum(lenl)
            dxl = []
            k = 0
            for i in range(0, len(strokex)):
                for j in range(0, len(strokex[i]) - 1):
                    dxl.append(1 / 3 * lenl[k] * (
                                (strokex[i][j + 1] - ux) ** 2 + (strokex[i][j] - ux) ** 2 + (strokex[i][j + 1] - ux) * (
                                    strokex[i][j] - ux)))
                    k += 1

            thx = np.sqrt(np.sum(dxl) / np.sum(lenl))
            m = 0
            for i in range(0, len(strokex)):
                for j in range(0, len(strokex[i])):
                    strokex[i][j] =  (strokex[i][j] - ux) / thx
                    strokey[i][j] = - (strokey[i][j] - uy) / thx
                    m += 1

            L = np.zeros((m - 1, 6), dtype=float)
            m = 0
            for i in range(0, len(strokex)):
                for j in range(0, len(strokex[i])):
                    if j != len(strokex[i]) - 1:
                        L[m, 0] = strokex[i][j]
                        L[m, 1] = strokey[i][j]
                        L[m, 2] = strokex[i][j + 1] - strokex[i][j]
                        L[m, 3] = strokey[i][j + 1] - strokey[i][j]
                        L[m, 4] = 1
                        L[m, 5] = 0
                    else:
                        if i != len(strokex) - 1:
                            L[m, 0] = strokex[i][j]
                            L[m, 1] = strokey[i][j]
                            L[m, 2] = strokex[i + 1][0] - strokex[i][j]
                            L[m, 3] = strokey[i + 1][0] - strokey[i][j]
                            L[m, 4] = 0
                            L[m, 5] = 1
                    m += 1

            A.append(L)
            LA.append(A)
            if total == show:
                fig_after_preprocess_plt = fig1.add_subplot(1, 3, 3)
                for i in range(0, len(strokex)):
                    fig_after_preprocess_plt.plot(strokex[i], strokey[i], color="black")
                plt.show()
            if total > show:
                break
            if struct.unpack("l", f.read(4))[0] != -1:
                print("error")
            total += 1

if not os.path.exists(ss.data_path):
    os.makedirs(ss.data_path)
with open(ss.data_path + "dla_total_" + str(ss.la_total) + "_dist_" + str(ss.la_remove_dist_th) + "_ang_" + str(ss.la_remove_ang_th),"wb") as f:
    pickle.dump(LA, f, protocol=4)
