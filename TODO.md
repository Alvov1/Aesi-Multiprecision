# CI Restructuring

## Target structure

| Workflow | Platforms | Compilers |
|---|---|---|
| `build.yml` | Ubuntu, macOS, Windows | GCC, Clang, MSVC |
| `test.yml` | Ubuntu | GCC (+ coverage/Codecov), Clang |
| `sanitize.yml` | Ubuntu, macOS | GCC, Clang |
| `cuda.yml` | Ubuntu | nvcc 12.6 ✓ done |
| `benchmarks.yml` | Ubuntu | — |

## Tasks

### build.yml
- [x] Rename/replace `integration.yml`
- [x] Same platforms and compilers as current integration (Ubuntu GCC, Ubuntu Clang, macOS Clang, Windows MSVC)
- [x] Expand compiler flags to match `test/`: `-Wall -Wextra -Werror -Wconversion -Wsign-conversion -Wpedantic -Wshadow -Wold-style-cast -Wnull-dereference -Wundef` for GCC/Clang, `/W4 /WX` for MSVC

### test.yml
- [x] Replace `coverage.yml`
- [x] Matrix: Ubuntu GCC (with `AESI_BUILD_COVERAGE=ON` + lcov + Codecov upload) and Ubuntu Clang (without coverage)
- [x] Keep GMP dependency (libgmp-dev)

### sanitize.yml
- [x] Rename `sanitize_multiple_platforms.yml` (name was misleading — only Ubuntu was actually used)
- [x] Add macOS runner alongside Ubuntu
- [x] Merge `sanitize/unsigned/` and `sanitize/signed/` into a single matrix job

### benchmarks.yml
- [ ] No changes needed
