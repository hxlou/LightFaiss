import sys
import os
import kp
import numpy as np

sys.path.insert(0, '/home/lhxx/LightFaiss/build/src/python')

import lightfaiss_py as lf

def test_flat_index():
    mgr = kp.Manager()

    # 创建一个 dim = 2 的向量
    index = lf.FlatIndex(2, 10, False, lf.MetricType.METRIC_INNER_PRODUCT, mgr)

    # 添加向量
    vectors = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    index.add(vectors)

    # 检索向量
    query = np.array([[1.0, 2.0]], dtype=np.float32)
    k = 2
    distances, indices = index.query_range(query, k, 0, 2, lf.DeviceType.CPU_BLAS)

    print("Distances:", distances)
    print("Indices:", indices)

if __name__ == "__main__":
    test_flat_index()
    print("Flat index test passed.")
    sys.exit(0)