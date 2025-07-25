# Fibonacci number calculation: a performance comparison across languages

This repository analyzes and compares the performance of calculating Fibonacci numbers using different programming languages and their respective runtimes and compilers. The goal is to determine the largest Fibonacci number that can be computed in under one second.

# Objective

The primary objective of this project is to benchmark the following environments:

- Python
    - CPython (the standard implementation)
    - PyPy (a high-performance just-in-time compiler)

- Rust
    - `build` (development mode)
    - `build --release` (optimized for performance)

# Prerequisites

Before you begin, ensure you have the following installed on your system:

- For Python
    - [**uv**](https://docs.astral.sh/uv/getting-started/installation/): An extremely fast Python package and project manager.
    - [**Poe the Poet**](https://poethepoet.natn.io/installation.html): A task runner for Python projects.

- For Rust
    - [**Rust**](https://www.rust-lang.org/tools/install): The Rust programming language and its package manager, Cargo.

# Getting started

**Installation**

- Python environment
    1. **Install uv:**
        On Linux, execute the following commands to install this fast package manager:
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
        exec $SHELL
        ```
    2. **Install Poe the Poet:**
        With `uv` installed, set up the task runner:
        ```bash
        uv tool install poethepoet
        ```

- Rust environment
    1. **Install Rust:**
        On Linux, run the following command and follow the on-screen instructions to install Rust and Cargo:
        ```bash
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        source $HOME/.cargo/env
        exec $SHELL
        ```

**Execution**

- Python
    - **CPython:** To run the tests using the standard Python implementation, use:
    ```bash
    poe run cpython
    ```

    - **PyPy:** To execute with the high-performance PyPy implementation, run:
    ```bash
    poe run pypy
    ```

- Rust
    - **Development build:** To compile and run in development mode, which is faster to compile but slower to execute:
    ```bash
    cargo build
    cargo run
    ```

    - **Release build:** For an optimized build that prioritizes runtime performance over compilation speed:
    ```bash
    cargo build --release
    cargo run --release
    ```

# Contributing

Contributions are welcome! If you would like to add a new language, an optimization, or improve the existing code, please feel free to open a pull request or an issue.
