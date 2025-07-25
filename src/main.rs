use num_bigint::BigUint;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use thousands::Separable;
use wait_timeout::ChildExt;

trait Fibonacci {
    fn call(&mut self, n: u64) -> BigUint;
    fn name(&self) -> &'static str;
}

struct SimpleRecursiveFibonacci;

impl Fibonacci for SimpleRecursiveFibonacci {
    fn call(&mut self, n: u64) -> BigUint {
        if n <= 1 {
            return n.into();
        }
        let a: BigUint = self.call(n - 1);
        let b: BigUint = self.call(n - 2);
        a + b
    }

    fn name(&self) -> &'static str {
        "simple_recursive_fibonacci"
    }
}

struct MemoizationRecursiveFibonacci {
    cache: HashMap<u64, BigUint>,
}

impl MemoizationRecursiveFibonacci {
    fn new() -> Self {
        let mut cache: HashMap<u64, BigUint> = HashMap::new();
        cache.insert(0, 0u64.into());
        cache.insert(1, 1u64.into());
        Self { cache }
    }
}

impl Fibonacci for MemoizationRecursiveFibonacci {
    fn call(&mut self, n: u64) -> BigUint {
        if let Some(value) = self.cache.get(&n) {
            return value.clone();
        }

        let last_known_n: u64 = self.cache.keys().max().cloned().unwrap_or(1);

        for i in (last_known_n + 1)..=n {
            let prev1: BigUint = self.cache.get(&(i - 1)).unwrap().clone();
            let prev2: BigUint = self.cache.get(&(i - 2)).unwrap().clone();
            self.cache.insert(i, prev1 + prev2);
        }

        self.cache.get(&n).unwrap().clone()
    }

    fn name(&self) -> &'static str {
        "memoization_recursive_fibonacci"
    }
}

struct IterativeFibonacci;

impl Fibonacci for IterativeFibonacci {
    fn call(&mut self, n: u64) -> BigUint {
        if n <= 1 {
            return n.into();
        }

        let mut a: BigUint = 0u64.into();
        let mut b: BigUint = 1u64.into();

        for _ in 1..n {
            let next: BigUint = a + &b;
            a = b;
            b = next;
        }
        b
    }

    fn name(&self) -> &'static str {
        "iterative_fibonacci"
    }
}

struct MatrixFibonacci;

impl Fibonacci for MatrixFibonacci {
    fn call(&mut self, n: u64) -> BigUint {
        fn matrix_multiplier(a: &[Vec<BigUint>], b: &[Vec<BigUint>]) -> Vec<Vec<BigUint>> {
            let mut result: Vec<Vec<BigUint>> = vec![
                vec![0u64.into(), 0u64.into()],
                vec![0u64.into(), 0u64.into()],
            ];

            for i in 0..2 {
                for j in 0..2 {
                    result[i][j] = (0..2).map(|k| &a[i][k] * &b[k][j]).sum();
                }
            }

            result
        }

        let mut base: Vec<Vec<BigUint>> = vec![
            vec![0u64.into(), 1u64.into()],
            vec![1u64.into(), 1u64.into()],
        ];

        let mut result: Vec<Vec<BigUint>> = vec![
            vec![1u64.into(), 0u64.into()],
            vec![0u64.into(), 1u64.into()],
        ];

        if n == 0 {
            return result[1][0].clone();
        }

        let mut new_n: u64 = n;

        while new_n > 0 {
            if new_n % 2 == 1 {
                result = matrix_multiplier(&result, &base);
            }

            base = matrix_multiplier(&base, &base);
            new_n /= 2;
        }

        result[1][0].clone()
    }

    fn name(&self) -> &'static str {
        "matrix_fibonacci"
    }
}

fn performance_tester(
    func: &mut dyn Fibonacci,
    time_limit_seconds: f64,
) -> (&'static str, u64, usize) {
    println!("--- Testing: ({}) ---", func.name());

    let time_execution = |n_val: u64| -> f64 {
        let self_exe: PathBuf = std::env::current_exe().expect("Failed to get current exe path");
        let mut child: Child = Command::new(self_exe)
            .args(["--run-worker", func.name(), &n_val.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn worker process");

        let start_time: Instant = Instant::now();
        let time_limit: Duration = Duration::from_secs_f64(time_limit_seconds);

        match child
            .wait_timeout(time_limit)
            .expect("Failed waiting for child")
        {
            Some(status) => {
                let elapsed: f64 = start_time.elapsed().as_secs_f64();
                if status.success() {
                    elapsed
                } else {
                    f64::INFINITY
                }
            }

            None => {
                child.kill().expect("Failed to kill child process");
                child.wait().expect("Failed waiting for killed child");
                f64::INFINITY
            }
        }
    };

    // Phase 1: Exponential search
    let increase_factor: f64 = 2.0;

    let mut n: u64 = 1;
    let mut last_n: u64 = n;
    let mut runtime: f64 = time_execution(n);

    while runtime < time_limit_seconds {
        last_n = n;
        n = (n as f64 * increase_factor).ceil() as u64;
        runtime = time_execution(n);
        print!("⬆️ Test {} - runtime ", n.separate_with_commas());

        if runtime <= 1.0 {
            println!("{runtime:.6}s");
        } else {
            println!(">1s");
        }
    }

    // Phase 2: Binary search
    let mut low: u64 = last_n;
    let mut high: u64 = n;
    let mut best_n: u64 = low;

    while low <= high {
        let mid: u64 = (low + high) / 2;
        if mid == 0 || mid == best_n {
            break;
        }

        let runtime: f64 = time_execution(mid);
        print!("⬆️ Test {} - runtime ", mid.separate_with_commas());

        if runtime <= 1.0 {
            println!("{runtime:.6}s");
        } else {
            println!(">1s");
        }

        if runtime < time_limit_seconds {
            best_n = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    let final_result: BigUint = func.call(best_n);
    let num_digits: usize = (final_result.bits() as f64 * 2.0f64.log10()).floor() as usize + 1;

    println!(
        "✅ The largest number calculated in less than {:.0}s was F({}), with {} digits.",
        time_limit_seconds,
        best_n.separate_with_commas(),
        num_digits.separate_with_commas()
    );
    println!("{}\n", "-".repeat(40));

    (func.name(), best_n, num_digits)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() == 4 && args[1] == "--run-worker" {
        let func_name: &String = &args[2];
        let n: u64 = args[3].parse().expect("Invalid number for n");

        match func_name.as_str() {
            "simple_recursive_fibonacci" => SimpleRecursiveFibonacci.call(n),
            "memoization_recursive_fibonacci" => MemoizationRecursiveFibonacci::new().call(n),
            "iterative_fibonacci" => IterativeFibonacci.call(n),
            "matrix_fibonacci" => MatrixFibonacci.call(n),
            _ => panic!("Unknown fibonacci function: {func_name}"),
        };
        return;
    }

    let mut fibonacci_functions: Vec<Box<dyn Fibonacci>> = vec![
        Box::new(SimpleRecursiveFibonacci),
        Box::new(MemoizationRecursiveFibonacci::new()),
        Box::new(IterativeFibonacci),
        Box::new(MatrixFibonacci),
    ];

    let mut results: HashMap<&'static str, (u64, usize)> = HashMap::new();

    for func in &mut fibonacci_functions {
        let (name, n, num_digits) = performance_tester(func.as_mut(), 1.0);
        results.insert(name, (n, num_digits));
    }

    let mut sorted_results: Vec<_> = results.into_iter().collect();
    sorted_results.sort_by(|a, b| b.1.0.cmp(&a.1.0));

    let max_size_k: usize = 1 + sorted_results
        .iter()
        .map(|(key, _)| key.len())
        .max()
        .unwrap_or(0);

    let max_size_n: usize = 4 + sorted_results
        .iter()
        .map(|(_, (n, _))| {
            let digits_minus_one: u64 = (*n as f64).log10().floor() as u64;
            digits_minus_one + (digits_minus_one / 3)
        })
        .max()
        .unwrap_or(0) as usize;

    println!("✅ Final results:");

    for (k, (n, d)) in &sorted_results {
        let n_formatted = format!("F({})", n.separate_with_commas());

        println!(
            "\t{:<k_width$}: {:<n_width$} with {} digits.",
            k,
            n_formatted,
            d.separate_with_commas(),
            k_width = max_size_k,
            n_width = max_size_n
        );
    }
}
