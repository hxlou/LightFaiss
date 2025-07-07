import sys

sys.path.insert(0, "/home/lhxx/LightFaiss/build/src/python")

import numpy as np
import lightfaiss.lightfaiss_py as lf
import lightfaiss.kp as kp

def test_flat_index_common():
    # 创建一个 dim = 2 的向量
    index = lf.FlatIndex(2, 10, False, lf.MetricType.METRIC_INNER_PRODUCT, None)

    # 添加向量
    # 每个向量长度为 2，把一个圆周分为100份作为实际的向量数据
    vectors = np.array([[np.cos(2 * np.pi * i / 100),
                        np.sin(2 * np.pi * i / 100)] for i in range(100)], dtype=np.float32)    
    index.add_vectors(vectors)
    
    if index.get_dim() != 2:
        raise ValueError("Dimension mismatch: expected 2, got {}".format(index.get_dim()))
    if index.get_num() != 100:
        raise ValueError("Number of vectors mismatch: expected 100, got {}".format(index.get_num()))
    if index.is_float16() != False:
        raise ValueError("Float16 mismatch: expected False, got {}".format(index.is_float16()))
    
    # 测试 save 和 load 方法
    index.save("test_flat_index.bin")
    index_loaded = lf.FlatIndex(2, 10, False, lf.MetricType.METRIC_INNER_PRODUCT, None)
    index_loaded.load("test_flat_index.bin")

    if index_loaded.get_dim() != index.get_dim():
        raise ValueError("Loaded dimension mismatch: expected {}, got {}".format(index.get_dim(), index_loaded.get_dim()))
    if index_loaded.get_num() != index.get_num():
        raise ValueError("Loaded number of vectors mismatch: expected {}, got {}".format(index.get_num(), index_loaded.get_num()))
    if index_loaded.is_float16() != index.is_float16():
        raise ValueError("Loaded Float16 mismatch: expected {}, got {}".format(index.is_float16(), index_loaded.is_float16()))
    for i in range(100):
        original_vector = index.reconstruct(i)
        loaded_vector = index_loaded.reconstruct(i)
        if not np.allclose(original_vector, loaded_vector):
            raise ValueError("Reconstructed vector mismatch at index {}: expected {}, got {}".format(i, original_vector, loaded_vector))


def test_flat_index_cpu():
    # 创建一个 dim = 2 的向量
    index = lf.FlatIndex(2, 10, False, lf.MetricType.METRIC_INNER_PRODUCT, None)

    # 添加向量
    # 每个向量长度为 2，把一个圆周分为100份作为实际的向量数据
    vectors = np.array([[np.cos(2 * np.pi * i / 100),
                        np.sin(2 * np.pi * i / 100)] for i in range(100)], dtype=np.float32)    
    index.add_vectors(vectors)

    # 新建查询向量，总共10个，把圆周分为10份
    query = np.array([[np.cos(2 * np.pi * i / 10),
                      np.sin(2 * np.pi * i / 10)] for i in range(10)], dtype=np.float32)
    k = 3
    indices, distances = index.query_range(query, k, 0, vectors.shape[0], lf.DeviceType.CPU)

    # 输出查询结果，包括index、换原向量、距离，每个查询三个都要输出，开头输出查询向量
    # 还原通过index.reconstruct方法
    for i in range(len(query)):
        print(f"Query vector {i}: {query[i]}")
        for j in range(k):
            idx = indices[i][j]
            distance = distances[i][j]
            original_vector = index.reconstruct(idx)
            print(f"  Index: {idx}, Original Vector: {original_vector}, Distance: {distance}")
        print("----------------------")

def test_flat_index_gpu():
    print("Testing FlatIndex on GPU...")

    mgr = kp.Manager()

    # 创建一个 dim = 2 的向量
    index = lf.FlatIndex(2, 10, False, lf.MetricType.METRIC_INNER_PRODUCT, mgr)

    # 添加向量
    # 每个向量长度为 2，把一个圆周分为100份作为实际的向量数据
    vectors = np.array([[np.cos(2 * np.pi * i / 100),
                        np.sin(2 * np.pi * i / 100)] for i in range(100)], dtype=np.float32)    
    index.add_vectors(vectors)

    # 新建查询向量，总共10个，把圆周分为10份
    query = np.array([[np.cos(2 * np.pi * i / 10),
                      np.sin(2 * np.pi * i / 10)] for i in range(10)], dtype=np.float32)
    k = 3
    indices, distances = index.query_range(query, k, 0, vectors.shape[0], lf.DeviceType.GPU)

    # 输出查询结果，包括index、换原向量、距离，每个查询三个都要输出，开头输出查询向量
    # 还原通过index.reconstruct方法
    for i in range(len(query)):
        print(f"Query vector {i}: {query[i]}")
        for j in range(k):
            idx = indices[i][j]
            distance = distances[i][j]
            original_vector = index.reconstruct(idx)
            print(f"  Index: {idx}, Original Vector: {original_vector}, Distance: {distance}")
        print("----------------------")

def test_L2_renorm_gpu():
    print("Testing GPU L2 renormalization...")

    mgr = kp.Manager()

    # 创建一个 dim = 2 的向量
    dim = 2
    nData = 1000
    vecs = np.array([[np.cos(2 * np.pi * i / nData) * 10.0,
                      np.sin(2 * np.pi * i / nData) * 10.0] for i in range(nData)], dtype=np.float32).flatten()
                      
    # 调用L2 renormalization
    lf.normalized_L2_gpu(mgr, vecs, dim, nData)

    # 验证结果
    isPassed = True
    for i in range(nData):
        x = vecs[i * 2 + 0]
        y = vecs[i * 2 + 1]
        gx = np.cos(2 * np.pi * i / nData)
        gy = np.sin(2 * np.pi * i / nData)
        if not (np.isclose(x, gx) and np.isclose(y, gy)):
            print(f"Renormalization failed at index {i}: expected ({gx}, {gy}), got ({x}, {y})")
            isPassed = False

    if isPassed:
        print("GPU L2 renormalization passed.")
    else:
        raise ValueError("GPU L2 renormalization failed.")

def test_L2_renorm_cpu():
    print("Testing CPU L2 renormalization on CPU...")

    # 创建一个 dim = 2 的向量
    dim = 2
    nData = 1000
    vecs = np.array([[np.cos(2 * np.pi * i / nData) * 10.0,
                      np.sin(2 * np.pi * i / nData) * 10.0] for i in range(nData)], dtype=np.float32).flatten()
                      
    # 调用L2 renormalization
    lf.normalized_L2_cpu(vecs, dim, nData)

    # 验证结果
    isPassed = True
    for i in range(nData):
        x = vecs[i * 2 + 0]
        y = vecs[i * 2 + 1]
        gx = np.cos(2 * np.pi * i / nData)
        gy = np.sin(2 * np.pi * i / nData)
        if not (np.isclose(x, gx) and np.isclose(y, gy)):
            print(f"Renormalization failed at index {i}: expected ({gx}, {gy}), got ({x}, {y})")
            isPassed = False

    if isPassed:
        print("CPU L2 renormalization on CPU passed.")
    else:
        raise ValueError("CPU L2 renormalization on CPU failed.")

if __name__ == "__main__":
    test_flat_index_cpu()
    test_flat_index_gpu()
    test_flat_index_common()
    test_L2_renorm_cpu()
    test_L2_renorm_gpu()
    print("[*FINAL*]: Flat index test passed.")
    sys.exit(0)