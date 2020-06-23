import numpy as np
import open3d as o3d
import os
import shutil

downsampled = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/'

total_error = []
faces = 300
results = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/averaged/' + \
          str(faces) + '_faces'
i = 0
error_sum = 0
vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 45]
total_error = np.zeros((1, len(vals)))
PC = 0
index = 0
for filename in os.listdir(results):
    if filename[0:5] == 'femur':
        femur_id = filename[len(filename)-11: len(filename)-4]
        real_load = o3d.io.read_point_cloud(downsampled + '/femur_bone_low_downsample_' + femur_id + '.ply')
        femur_real = np.asarray(real_load.points)
        i += 1
        if i == 6:
            if filename[17] == '_':
                PC = filename[16]
            else:
                PC = filename[16:18]
            index = vals.index(int(PC))

            averaged_error = error_sum/5
            total_error[0, index] = averaged_error
            i = 0
            error_sum = 0

        prediction_load = o3d.io.read_point_cloud(results + '/' + filename)
        prediction = np.asarray(prediction_load.points)
        x1 = np.transpose(prediction)  # converts femur_target into 3x54 - f1, f2, f3, f4, ... f54
        x2 = np.transpose(femur_real)  # converts femur_target into 3x1995 - m1, m2, m3, m4, ... m1995
        x3 = np.repeat(x1, x2.shape[1], axis=1)  # creates f1, f1, f1, ... f2, f2, f2, ... f3, f3, f3, ...
        x4 = np.tile(x2, x1.shape[1])  # creates m1, m2, m3, m4, ... m1, m2, m3, m4, ... m1, m2, m3, m4, ...
        distances = np.linalg.norm(np.transpose(x3) - np.transpose(x4), axis=1).reshape(1995, -1)  # normal dist
        # combination of f and m, and reshapes the matrix into [[(f1-m1), (f1-m2), (f1-m3), ...], [(f2-m1), (f2-m2), (f2-m3), ...], [(f3-m1), (f3-m2), (f3-m3), ...], ...]
        min_dist = np.min(distances, axis=1)
        min_dist2 = np.min(distances, axis=0)
        # error.append((np.sum(min_dist)+np.sum(min_dist2)) / (prediction.shape[0]+femur_real.shape[0]))
        error_sum += np.power((np.sum(np.power(min_dist, 2)) + np.sum(np.power(min_dist2, 2))) / (
                prediction.shape[0] + femur_real.shape[0]), 0.5)  # RMSD Error

np.save('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/averaged/' + str(faces) +
        '_faces/total_averaged_RMSD_error.npy', np.array(total_error))

print('done')
