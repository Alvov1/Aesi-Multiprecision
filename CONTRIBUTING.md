# Contributing

## Commit messages

Commit messages follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`, `build`, `perf`, `revert`.

Examples:
```
feat: add modular inverse operation
fix: correct overflow in multiplication for 64-bit blocks
build: pin googletest to v1.17.0
```

Pull request titles must also follow this format — they are checked automatically on every PR.

## Branches

Use `feature/`, `fix/`, `docs/`, `ci/`, `refactor/` prefixes:
```
feature/modular-inverse
fix/multiplication-overflow
docs/update-readme
```

## Running tests

Requires [GMP](https://gmplib.org/) (`libgmp-dev` / `brew install gmp`).

```bash
cmake -B build -DAESI_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

## Running benchmarks

Requires GMP, [Google Benchmark](https://github.com/google/benchmark), and [Crypto++](https://www.cryptopp.com/).

```bash
cmake -B build -DAESI_BUILD_BENCHMARKS=ON
cmake --build build
./build/benchmark/Benchmarking --benchmark_format=json
```

## Running sanitizers

```bash
cmake -B build -DAESI_BUILD_SANITIZERS=ON
cmake --build build
./build/sanitize/unsigned/AesiSanitize
```

## Opening a PR

Push your branch and open a PR via the GitHub interface. Keep each PR focused on a single concern.
