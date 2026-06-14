# ALIEN — Repository Instructions

ALIEN is an artificial life simulation built on a 2D CUDA particle engine for soft
bodies and fluids. Mainly C++23 and CUDA, built with CMake + vcpkg (manifest mode).
An NVIDIA CUDA GPU is required for engine functionality and the engine tests.
The GUI uses Dear ImGui.

## Language

Converse in the language the user used, but **everything that is committed, pushed,
or created on GitHub must be in English**: commit messages, branch slugs, PR titles
and bodies, code comments, identifiers. Keep commit messages short and imperative.

## Code style

- 4 spaces, no tabs
- Allman braces
- camelCase for variables and functions
- PascalCase for classes
- UPPER_SNAKE_CASE for constants
- `.h` for C++ headers, `.cuh` for CUDA headers, `.cpp` / `.cu` for implementations
- Avoid unnecessary comments; prefer self-documenting code
- Prefer `.at()` over `[]` for `std::vector` access unless there is a strong local reason

## Hard rules

- Do **not** run CodeQL / `codeql_checker` / any CodeQL security scanning
- Do **not** run `git submodule update`
- `external/vcpkg` is a pinned submodule — never modify or commit it. If it shows as
  modified, restore with `git restore external/vcpkg`. Never `git add external/vcpkg`.
- Do not cancel long-running builds or tests; they can take several minutes.

## Build (Windows)

Build with the repo-root script, not the default Visual Studio generator:

```
build-windows-ninja.bat          # Release (default)
build-windows-ninja.bat Debug    # Debug
```

It sets up MSVC via vcvars64 and uses the "Ninja Multi-Config" CMake preset
(`cmake --preset ninja` + `cmake --build --preset ninja-release`), compiling the CUDA
translation units in parallel. Executables land under **`build-ninja\Release\`**
(e.g. `alien.exe`, `cli.exe`, `EngineTests.exe`) — not the older `build\Release\`.

A struct / constant-memory / kernel `.cuh` change needs a clean rebuild, otherwise
stale kernels linger and weak tests can pass against old code.

## Tests

Executables under `build-ninja\Release\`:

```
EngineInterfaceTests.exe   (<1s)
NetworkTests.exe           (<1s)
PersisterTests.exe         (~1.4s)
EngineTests.exe            (~150s — needs the GPU; do not cancel)
```

GUI-only changes under `source/Gui/` that do not touch engine, network, persistence,
CLI, or shared code do not strictly need tests, but still build. For a targeted CUDA
failure: `EngineTests.exe -d --gtest_filter=Suite.Test` (debug mode is much slower —
use it for single tests only, not the full suite).

## Formatting

The clang-format **version matters**: format with **19.1.5** (bundled with Visual
Studio 2022 Community — bare `clang-format` on PATH resolves to it). Do **not** use the
VS Insiders v22 binary — same `source/_clang-format` config, but it reflows lines
differently and creates churn. Format only the files you modified, and only if they are
already clean at HEAD:

```
clang-format --style=file:source/_clang-format --dry-run --Werror <file>
```

Several committed files are not clang-format-clean, so running a whole-file format on
them reflows unrelated lines. ColumnLimit is 160; clang-format does not always wrap long
builder-chain assignments beyond 160, and that is accepted.

## Layout

```
source/Base/                 Common utilities, math, logging
source/Cli/                  Command-line interface
source/EngineGpuKernels/     CUDA kernels
source/EngineImpl/           CPU-side engine implementation
source/EngineInterface/      Abstract simulation APIs
source/EngineInterfaceTests/ EngineInterface unit tests
source/EngineTests/          CUDA engine integration tests
source/Gui/                  Dear ImGui GUI
source/Network/              HTTP / cloud features
source/PersisterImpl/        File I/O and serialization
external/                    Third-party dependencies incl. the pinned vcpkg submodule
resources/                   Runtime assets
```
