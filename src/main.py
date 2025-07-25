from math import ceil, floor, inf, log10
from multiprocessing import Process
from statistics import median
from time import perf_counter
from typing import Callable


def simple_recursive_fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return simple_recursive_fibonacci(n - 1) + simple_recursive_fibonacci(n - 2)


def memoization_recursive_fibonacci(n: int) -> int:
    cache: dict[int, int] = {0: 0, 1: 1}
    last_known_n: int = max(cache.keys())

    for i in range(last_known_n + 1, n + 1):
        prev1: int = cache[i - 1]
        prev2: int = cache[i - 2]
        cache[i] = prev1 + prev2

    return cache.get(n, 0)


def iterative_fibonacci(n: int) -> int:
    if n <= 1:
        return n

    a: int = 0
    b: int = 1
    for _ in range(n - 1):
        a, b = b, a + b

    return b


def matrix_fibonacci(n: int) -> int:
    def matrix_multiplier(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
        result: list[list[int]] = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    result[i][j] += a[i][k] * b[k][j]

        return result

    base: list[list[int]] = [[1, 1], [1, 0]]
    result: list[list[int]] = [[1, 0], [0, 1]]

    n_power = n
    while n_power > 0:
        if n_power % 2 == 1:
            result = matrix_multiplier(result, base)

        base = matrix_multiplier(base, base)
        n_power //= 2

    return result[1][0]


def performance_tester(
    func: Callable[[int], int],
    num_runs: int = 5,
    increase_factor: float = 2.0,
    time_limit_seconds: float = 1.0,
) -> tuple[str, int, int]:
    print(f"--- Testing: ({func.__name__}) ---")

    def time_execution(n_val: int) -> float:
        runtimes: list[float] = []
        for _ in range(num_runs):
            process = Process(target=func, args=(n_val,))
            start_time: float = perf_counter()
            process.start()
            process.join(timeout=time_limit_seconds)
            end_time = perf_counter()

            if process.is_alive():
                process.terminate()
                process.join()
                continue

            if process.exitcode == 0:
                runtimes.append(end_time - start_time)

        if not runtimes:
            return inf

        return median(runtimes)

    # Phase 1: Exponential search
    n: int = 1
    last_n: int = n
    median_runtime: float = time_execution(n)

    while median_runtime < time_limit_seconds:
        last_n = n
        n = ceil(n * increase_factor)
        median_runtime = time_execution(n)
        print(f"⬆️ Test {n:,} - median runtime", end=" ")

        if median_runtime < time_limit_seconds:
            print(f"{median_runtime:,.6f}s.")
        else:
            print(f">{time_limit_seconds:,.6f}s.")

    # Phase 2: Binary search
    low: int = last_n
    high: int = n
    best_n: int = low

    while low <= high:
        mid: int = (low + high) // 2
        if mid == 0 or mid == best_n:
            break

        median_runtime = time_execution(mid)
        print(f"⬆️ Test {mid:,} - median runtime", end=" ")

        if median_runtime < time_limit_seconds:
            best_n = mid
            low = mid + 1
            print(f"{median_runtime:,.6f}s.")
        else:
            high = mid - 1
            print(f">{time_limit_seconds:,.6f}s.")

    final_result = func(best_n)
    num_digits = floor(final_result.bit_length() * log10(2)) + 1
    print(
        f"✅ The largest number calculated in less than {time_limit_seconds:,.6f}s was F({best_n:,}), with {num_digits:,} digits."
    )
    print("-" * 40 + "\n")

    return func.__name__, best_n, num_digits


if __name__ == "__main__":
    fibonacci_functions: list[Callable[[int], int]] = [
        simple_recursive_fibonacci,
        memoization_recursive_fibonacci,
        iterative_fibonacci,
        matrix_fibonacci,
    ]
    results: dict[str, tuple[int, int]] = dict()

    for func in fibonacci_functions:
        name, n, num_digits = performance_tester(func)
        results.update({name: (n, num_digits)})

    results = dict(sorted(results.items(), key=lambda x: x[1][0], reverse=True))
    if not results:
        print("No functions completed the benchmark.")
        exit()
    max_size_k: int = 1 + max(map(len, results.keys()))
    max_size_n: int = 4 + max(
        [
            floor(log10(v[0])) + floor(log10(v[0])) // 3
            for v in results.values()
            if v[0] > 0
        ],
        default=0,
    )

    print("✅ Final results:")
    for k, (n, d) in results.items():
        n_formatted = f"F({n:,})"
        print(f"\t{k:<{max_size_k}}: {n_formatted:<{max_size_n}} with {d:,} digits.")
