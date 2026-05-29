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
- [ ] Rename/replace `integration.yml`
- [ ] Same platforms and compilers as current integration (Ubuntu GCC, Ubuntu Clang, macOS Clang, Windows MSVC)
- [ ] Expand compiler flags to match `test/`: `-Wall -Wextra -Werror -Wconversion -Wsign-conversion -Wpedantic -Wshadow -Wold-style-cast -Wnull-dereference -Wundef` for GCC/Clang, `/W4 /WX` for MSVC

### test.yml
- [ ] Replace `coverage.yml`
- [ ] Matrix: Ubuntu GCC (with `AESI_BUILD_COVERAGE=ON` + lcov + Codecov upload) and Ubuntu Clang (without coverage)
- [ ] Keep GMP dependency (libgmp-dev)

### sanitize.yml
- [ ] Rename `sanitize_multiple_platforms.yml` (name was misleading — only Ubuntu was actually used)
- [ ] Add macOS runner alongside Ubuntu
- [ ] Merge `sanitize/unsigned/` and `sanitize/signed/` into a single matrix job

### benchmarks.yml
- [ ] No changes needed
