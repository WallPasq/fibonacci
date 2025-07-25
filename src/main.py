from math import ceil, floor, inf, log10
from multiprocessing import Process
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
        prev1: int = cache.get(i - 1, 0)
        prev2: int = cache.get(i - 2, 1)
        cache.update({i: prev1 + prev2})

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
    def matrix_multiplier(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
        result: list[list[int]] = [[0 for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    result[i][j] += A[i][k] * B[k][j]

        return result

    base: list[list[int]] = [[0, 1], [1, 1]]
    result: list[list[int]] = [[int(i == j) for i in range(2)] for j in range(2)]

    if n == 0:
        return result[1][0]

    while n > 0:
        if n % 2 == 1:
            result = matrix_multiplier(result, base)

        base = matrix_multiplier(base, base)
        n //= 2

    return result[1][0]


def performance_tester(
    func: Callable[[int], int], time_limit_seconds: float = 1.0
) -> tuple[str, int, int]:
    print(f"--- Testing: ({func.__name__}) ---")

    def time_execution(n_val: int) -> float:
        process = Process(target=func, args=(n_val,))
        start_time: float = perf_counter()
        process.start()
        process.join(timeout=time_limit_seconds)
        end_time = perf_counter()

        if process.is_alive():
            process.terminate()
            process.join()
            return inf

        if process.exitcode != 0:
            return inf

        return end_time - start_time

    # Phase 1: Exponential search
    increase_factor: float = 2.0

    n: int = 1
    last_n: int = n
    runtime: float = time_execution(n)

    while runtime < time_limit_seconds:
        last_n = n
        n = ceil(n * increase_factor)
        runtime = time_execution(n)
        print(f"⬆️ Test {n:,} - runtime", end=" ")

        if runtime <= 1.0:
            print(f"{runtime:,.6f}.")
        else:
            print(">1s.")

    # Phase 2: Binary search
    low: int = last_n
    high: int = n
    best_n: int = low

    while low <= high:
        mid: int = (low + high) // 2
        if mid == 0 or mid == best_n:
            break

        runtime = time_execution(mid)
        print(f"⬆️ Test {mid:,} - runtime", end=" ")

        if runtime <= 1.0:
            print(f"{runtime:,.6f}.")
        else:
            print(">1s.")

        if runtime < time_limit_seconds:
            best_n = mid
            low = mid + 1
        else:
            high = mid - 1

    final_result = func(best_n)
    num_digits = floor(final_result.bit_length() * log10(2)) + 1
    print(
        f"✅ The largest number calculated in less than {time_limit_seconds:,.0f}s was F({best_n:,}), with {num_digits:,} digits."
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
    max_size_k: int = 1 + max(map(lambda x: len(x[0]), results.items()))
    max_size_n: int = 4 + max(
        map(lambda x: x + x // 3, map(lambda x: floor(log10(x[1][0])), results.items()))
    )

    print("✅ Final results:")
    for k, (n, d) in results.items():
        n_formatted = f"F({n:,})"
        print(f"\t{k:<{max_size_k}}: {n_formatted:<{max_size_n}} with {d:,} digits.")
