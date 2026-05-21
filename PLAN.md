# Refactoring Plan

## 1. ~~Перенести заголовки в `include/AesiMultiprecision/`~~ ✓
## 2. ~~Интегрировать тесты, бенчмарки и sanitizers в корневой CMake~~ ✓
## 2а. ~~Добавить версию и install rules в CMakeLists.txt~~ ✓

## 3. ~~Упростить `ci/` — убрать дублирование~~ ✓

## 4. Добавить версию и install rules в CMakeLists.txt
**Приоритет: средний** — нужно для `find_package` и `cmake --install`

- Добавить `VERSION` в `project()`
- Добавить `install(TARGETS ...)` и `install(DIRECTORY include/ ...)`
- Сгенерировать `AesiMultiprecisionConfig.cmake` через `CMakePackageConfigHelpers`

## 5. ~~Удалить `sanitize/signed/` или добавить в CI~~ ✓
## 6. ~~Убрать `sanitize/primes.bin` из git~~ ✓