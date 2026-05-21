# Refactoring Plan

## 1. Перенести заголовки в `include/AesiMultiprecision/`
**Приоритет: высокий** — влияет на потребителей библиотеки

- Создать `include/AesiMultiprecision/Aeu.h` и `Aesi.h`
- Обновить корневой `CMakeLists.txt`: `target_include_directories(... INTERFACE include/)`
- Обновить `ci/build/`, `ci/integration/`, `test/`, `sanitize/`, `benchmark/` — везде поменять пути include
- Обновить README с новым способом подключения

## 2. Интегрировать тесты, бенчмарки и sanitizers в корневой CMake
**Приоритет: высокий** — сейчас нельзя собрать всё одной командой

- Добавить опции `AESI_BUILD_TESTS`, `AESI_BUILD_BENCHMARKS`, `AESI_BUILD_SANITIZERS`
- Подключить `add_subdirectory(test)`, `add_subdirectory(benchmark)`, `add_subdirectory(sanitize/unsigned)` под флагами
- Добавить `enable_testing()` и `include(CTest)` в корень
- Убрать дублирование CMake-логики между `test/` и `ci/`

## 3. Упростить `ci/` — убрать дублирование
**Приоритет: средний**

- `ci/build/` — smoke-тест "компилируется ли как библиотека". Оставить.
- `ci/integration/` — по сути тест, который дублирует `test/`. Рассмотреть объединение с `test/` или явно разграничить что тестирует каждый.

## 4. Добавить версию и install rules в CMakeLists.txt
**Приоритет: средний** — нужно для `find_package` и `cmake --install`

- Добавить `VERSION` в `project()`
- Добавить `install(TARGETS ...)` и `install(DIRECTORY include/ ...)`
- Сгенерировать `AesiMultiprecisionConfig.cmake` через `CMakePackageConfigHelpers`

## 5. ~~Удалить `sanitize/signed/` или добавить в CI~~ ✓
## 6. ~~Убрать `sanitize/primes.bin` из git~~ ✓