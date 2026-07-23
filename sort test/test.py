# import torch
import numpy as np
import random
import time


class Timer:
    def __init__(self, text=""):
        self.text = text
        self.start_time = time.perf_counter()

    def reset(self):
        """
        重置计时器的起始时间，将开始时间设为当前时间。

        用于重新开始计时，常用于需要重新测量时间间隔的场景。
        """

        self.start_time = time.perf_counter()

    def auto_unit_transform(self, time):
        units = ["s", "ms", "us", "ns"]
        unit_index = 0
        while time < 1:
            time *= 1000
            unit_index += 1
        return f"{time:.6f} {units[unit_index]}"

    def elapsed(self):
        elapsed_time = time.perf_counter() - self.start_time
        print(f"{self.text}: {self.auto_unit_transform(elapsed_time)}")
        return elapsed_time

    def __call__(self, func):
        """
        支持将 Timer 实例直接用作装饰器，例如：

            @Timer("排序耗时")
            def my_sort(arr):
                ...

        被装饰的函数被调用时，会自动打印并累加本次调用的耗时。
        """

        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            cost = time.perf_counter() - start
            label = self.text or getattr(func, "__name__", "func")
            print(f"{label}: {self.auto_unit_transform(cost)}")
            return result

        return wrapper

    @classmethod
    def timer(cls, func=None, *, text=""):
        """
        类方法装饰器，支持两种用法：

            @Timer.timer
            def my_sort(arr):
                ...

            @Timer.timer(text="排序耗时")
            def my_sort(arr):
                ...

        被装饰函数的函数名（或指定的 text）会作为打印标签。
        """

        import functools

        def decorator(fn):
            label = text or getattr(fn, "__name__", "func")

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                cost = time.perf_counter() - start
                print(f"{label}: {cls().auto_unit_transform(cost)}")
                return result

            return wrapper

        if func is not None:
            return decorator(func)
        return decorator


@Timer.timer(text="Python内置排序耗时")
def builtin_sort(arr):
    """
    使用 Python 内置的排序函数进行排序。

    参数:
        arr (list): 待排序的数组。

    返回:
        list: 排序后的数组。
    """
    return sorted(arr)


@Timer.timer(text="Python实现快速排序耗时")
def pythonic_sort(arr):
    """
    使用 Python 实现的快速排序函数

    参数:
        arr (list): 待排序的数组。

    返回:
        list: 排序后的数组。
    """

    stack = [(arr, False)]
    result = []
    while stack:
        current, processed = stack.pop()
        if len(current) <= 1:
            result.extend(current)
            continue
        if not processed:
            pivot = current[len(current) // 2]
            left = [x for x in current if x < pivot]
            middle = [x for x in current if x == pivot]
            right = [x for x in current if x > pivot]
            stack.append((right, False))
            stack.append((middle, True))
            stack.append((left, False))
        else:
            result.extend(current)
    return result
@Timer.timer(text="Python实现选择排序耗时")
def choose_sort(arr):
    """
    使用 Python 实现的选择排序函数

    参数:
        arr (list): 待排序的数组。

    返回:
        list: 排序后的数组。
    """
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr


@Timer.timer(text="CPU矩阵排序耗时")
def cpu_matrix_sort(arr):
    """

    参数:
        arr (list of list): 待排序的向量。

    返回:
        list of list: 排序后的向量。
    """

    v = np.array(arr, dtype=np.int32)  # 构造向量
    v_dim = v.shape[0]  # 获取向量的维度

    # Step1: 构造差值矩阵D_ij
    D1 = v.repeat(len(v)).reshape(v_dim, v_dim)  # 重复向量以构造差值矩阵的被减数
    D2 = D1.T  # 转置差值矩阵的减数
    D_ij = D1 - D2  # 计算差值矩阵

    # Step2: 利用阶跃函数获得S矩阵
    S = np.where(D_ij >= 0, 1, 0)  # 直接通过比较，计算S矩阵

    # Step3: 构造变换矩阵P_ij
    # 计算每一行的和r_i
    r_i = np.sum(S, axis=1)  # 计算每一行的和r
    P = np.zeros_like(S)  # 初始化变换矩阵P_ij
    P[np.arange(v_dim), r_i - 1] = (
        1  # 原构造方法即将每一行的第r_i个位置(索引-1)设为1，其余位置设为0
    )

    # Step4: 对向量v进行变换，得到排序后的列向量T
    t = v @ P  # 矩阵乘法得到排序后的列向量T
    t = t.tolist()
    # T.reverse()
    return t

import torch
@Timer.timer(text="GPU矩阵排序耗时")
@torch.no_grad()
def gpu_matrix_sort(arr):
    """
    GPU矩阵排序算法

    参数:
        arr (list of list): 待排序的向量。

    返回:
        list of list: 排序后的向量。
    """

    v = torch.tensor(arr, dtype=torch.int32)  # 构造向量
    v_dim = v.shape[0]  # 获取向量的维度

    # Step1: 构造差值矩阵D_ij
    D1 = v.repeat(len(v)).reshape(v_dim, v_dim)  # 重复向量以构造差值矩阵的被减数
    D2 = D1.T  # 转置差值矩阵的减数
    D_ij = D1 - D2  # 计算差值矩阵

    # Step2: 利用阶跃函数获得S矩阵
    S = (D_ij >= 0).int()  # 直接通过比较，计算S矩阵（保持与 v 一致的 int32）

    # Step3: 构造变换矩阵P_ij
    r_i = torch.sum(S, dim=1)  # 计算每一行的和r_i
    P = torch.zeros_like(S)  # 初始化变换矩阵P_ij
    P[torch.arange(v_dim), r_i - 1] = (
        1  # 原构造方法即将每一行的第r_i个位置(索引-1)设为1，其余位置设为0
    )
    # Step4: 对向量v进行变换，得到排序后的列向量T
    # 矩阵乘法统一在 float32 下进行，结果转回 int32
    t = (v.float() @ P.float()).int()  # 矩阵乘法得到排序后的列向量T
    t = t.tolist()
    t.reverse()
    return t

@Timer.timer(text="分块CPU矩阵排序耗时(chunk_size=8192)")
def chunk_wise_cpu_matrix_sort(arr):
    """
    分块CPU矩阵排序算法，避免内存溢出

    参数:
        arr (list of list): 待排序的向量。
        chunk_size (int): 每个块的大小。
    """
    chunk_size=8192
    v = np.array(arr, dtype=np.int32)  # 构造向量
    v_dim = v.shape[0]  # 获取向量的维度
    t = np.zeros_like(v)  # 初始化排序后的向量
    for i in range(0, len(arr), chunk_size):
        if i + chunk_size <= len(arr): # 标准块
            # Step1: 构造块内差值矩阵D_chunk_ij
            D1_chunk = v[i : i + chunk_size].repeat(v_dim).reshape(chunk_size, v_dim)
            D2_chunk = np.tile(v, (chunk_size, 1))  # 使用np.tile重复向量以构造差值矩阵的减数
            D_chunk_ij = D1_chunk - D2_chunk  # 计算块内差值矩阵
            real_chunk_size = chunk_size
        else: # 处理不完整的最后一块
            # Step1: 构造块内差值矩阵D_chunk_ij
            real_chunk_size = len(arr) - i # 计算最后一块的实际大小
            D1_chunk = v[i :].repeat(v_dim).reshape(real_chunk_size, v_dim)
            D2_chunk = np.tile(v, (real_chunk_size, 1))  # 使用np.tile重复向量以构造差值矩阵的减数
            D_chunk_ij = D1_chunk - D2_chunk  # 计算块内差值矩阵
        
        # 后续操作相同
        # Step2: 利用阶跃函数获得S矩阵
        S_chunk = np.where(D_chunk_ij >= 0, 1, 0)  # 直接通过比较，计算S矩阵

        # Step3: 构造变换矩阵P_ij
        # 计算块内每一行的和r_i
        r_chunk_i = np.sum(S_chunk, axis=1)  # 计算每一行的和r
        P_chunk = np.zeros_like(S_chunk)  # 初始化变换矩阵P_ij
        P_chunk[np.arange(real_chunk_size), r_chunk_i - 1] = 1  # 原构造方法即将每一行的第r_i个位置(索引-1)设为1，其余位置设为0

        # Step4: 利用分块矩阵乘法对向量v进行变换，得到排序后的列向量T
        t = t + P_chunk.T @ v[i:i+real_chunk_size]  # 矩阵乘法得到排序后的列向量T，稀疏矩阵可直接相加
    t = t.tolist()
    return t

@Timer.timer(text="分块GPU矩阵排序耗时(chunk_size=4096)")
@torch.no_grad()
def chunk_wise_gpu_matrix_sort(arr, device="cuda"):
    """
    分块GPU矩阵排序算法（直接移植NumPy逻辑版，未做优化）

    参数:
        arr (list or torch.Tensor): 待排序的向量。
        chunk_size (int): 每个块的大小。
        device (str): 计算设备，默认为 'cuda'。
    """
    chunk_size=4096 
    # 构造向量并移至GPU
    v = torch.tensor(arr, dtype=torch.int32, device=device)
    v_dim = v.shape[0]  # 获取向量的长度（原代码称为维度）
    t = torch.zeros_like(v)  # 初始化排序后的向量
    
    for i in range(0, len(arr), chunk_size):
        if i + chunk_size <= len(arr): # 标准块
            real_chunk_size = chunk_size
            v_chunk = v[i : i + chunk_size]
            
            # Step1: 构造块内差值矩阵D_chunk_ij
            # 注意：NumPy的 repeat 对应 PyTorch 的 repeat_interleave
            D1_chunk = v_chunk.repeat_interleave(v_dim).reshape(real_chunk_size, v_dim)
            # 注意：NumPy的 tile 对应 PyTorch 的 unsqueeze(0) + repeat
            D2_chunk = v.unsqueeze(0).repeat(real_chunk_size, 1)  
            D_chunk_ij = D1_chunk - D2_chunk
            
        else: # 处理不完整的最后一块
            real_chunk_size = len(arr) - i
            v_chunk = v[i :]
            
            # Step1: 构造块内差值矩阵D_chunk_ij
            D1_chunk = v_chunk.repeat_interleave(v_dim).reshape(real_chunk_size, v_dim)
            D2_chunk = v.unsqueeze(0).repeat(real_chunk_size, 1)
            D_chunk_ij = D1_chunk - D2_chunk
        
        # 后续操作相同
        # Step2: 利用阶跃函数获得S矩阵
        # np.where 对应 torch.where，为了保持int64类型，显式传入张量
        S_chunk = torch.where(
            D_chunk_ij >= 0, 
            torch.tensor(1, dtype=torch.int32, device=device), 
            torch.tensor(0, dtype=torch.int32, device=device)
        )

        # Step3: 构造变换矩阵P_ij
        # 计算块内每一行的和r_i
        r_chunk_i = torch.sum(S_chunk, dim=1)
        P_chunk = torch.zeros_like(S_chunk)
        
        # 高级索引赋值 (对应 NumPy 的 P_chunk[np.arange(...), ...] = 1)
        P_chunk[torch.arange(real_chunk_size, device=device), r_chunk_i - 1] = 1

        # Step4: 利用分块矩阵乘法对向量v进行变换，得到排序后的列向量T
        # P_chunk.T 形状为 (v_dim, real_chunk_size)，v_chunk 形状为 (real_chunk_size,)
        # 矩阵乘法结果形状为 (v_dim,)，与 t 形状一致可直接相加
        t = t + (P_chunk.T.float() @ v_chunk.float()).int()
        
    # 移回 CPU 并转为 list
    t = t.cpu().tolist()
    
    return t

def check_sorted(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True

if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("开始排序测试，列表长度70000")
    # 生成一个包含100000个随机整数的数组
    arr = list(range(70_000))
    random.shuffle(arr)

    # 测试 Python 内置排序函数
    sorted_builtin = builtin_sort(arr.copy())

    sorted_pythonic = pythonic_sort(arr.copy())
    sorted_chunk_gpu_matrix = chunk_wise_gpu_matrix_sort(arr.copy())
    
    sorted_choose = choose_sort(arr.copy())
    # sorted_chunk_cpu_matrix = chunk_wise_cpu_matrix_sort(arr.copy())
    try:
        # 测试 CPU 矩阵排序函数
        sorted_cpu_matrix = cpu_matrix_sort(arr.copy())
    except MemoryError:
        print("CPU矩阵排序耗时: 内存不足")

    try:
        sorted_gpu_matrix = gpu_matrix_sort(arr.copy())
    except (MemoryError, RuntimeError):
        print("GPU矩阵排序耗时: 内存不足")
    

    # 验证两种排序方法的结果是否一致
    assert check_sorted(sorted_builtin), "Python内置排序结果不正确！"
    assert check_sorted(sorted_pythonic), "Python实现快速排序结果不正确！"
    # assert check_sorted(sorted_chunk_cpu_matrix), "分块CPU矩阵排序结果不正确！"
    # assert check_sorted(sorted_cpu_matrix), "CPU矩阵排序结果不正确！"
    # assert check_sorted(sorted_gpu_matrix), "GPU矩阵排序结果不正确！"
    assert check_sorted(sorted_chunk_gpu_matrix), "分块GPU矩阵排序结果不正确！"
    # assert check_sorted(sorted_choose), "选择排序结果不正确！"