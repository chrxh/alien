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
- `build-ninja` / `build-ninja-hip` belong to the IDE. Never configure into them,
  build into them or delete them — Visual Studio may be open on this folder and
  only recovers by deleting `.vs` and restarting. Agents build into `build-agent`
  (see below); a clean rebuild means deleting `build-agent`.

## Build (Windows)

Build with the repo-root script, not the default Visual Studio generator. Agents
always pass `agent`:

```
build-windows-ninja.bat agent          # Release (default)
build-windows-ninja.bat agent Debug    # Debug
```

It sets up MSVC via vcvars64 and uses the "Ninja Multi-Config" CMake preset,
compiling the CUDA translation units in parallel.

There are two separate build trees:

| Caller | Preset | Output |
| --- | --- | --- |
| Visual Studio, or the script run by hand | `ninja` | `build-ninja\Release\` |
| Agents (`agent` argument) | `ninja-agent` | `build-agent\Release\` |

Visual Studio configures the `ninja` preset into `build-ninja` and caches its model
of that tree under `.vs`. An outside build regenerates the tree, invalidates that
cache and breaks the next build in the IDE until `.vs` is deleted — hence the second
tree. Without an argument the script uses the IDE tree. Only Claude Code (`CLAUDECODE`
is set) and `ALIEN_BUILD_TREE=agent` switch to the agent tree automatically, so every
other agent needs the explicit `agent` argument.

Executables (`alien.exe`, `alien-cli.exe`, `EngineTests.exe`) land under the `Release\`
subdirectory of the respective tree — not the older `build\Release\`.

A struct / constant-memory / kernel `.cuh` change needs a clean rebuild, otherwise
stale kernels linger and weak tests can pass against old code.

## Tests

Executables under `build-agent\Release\` (`build-ninja\Release\` for a manual build):

```
BaseTests.exe              (<1s)
DataTests.exe              (<1s)
EngineInterfaceTests.exe   (<1s)
NetworkTests.exe           (<1s)
PersisterTests.exe         (~1.4s)
EngineTests.exe            (~150s — needs the GPU; do not cancel)
```

GUI-only changes under `source/Gui/` that do not touch engine, network, persistence,
CLI, or shared code do not strictly need tests, but still build. For a targeted CUDA
failure: `EngineTests.exe -d --gtest_filter=Suite.Test` (debug mode is much slower —
use it for single tests only, not the full suite).

## Build and tests on Linux

On Linux, for example in the GitHub Copilot cloud agent environment that
`.github/workflows/copilot-setup-steps.yml` prepares, the build tree is `build` and
the executables land directly in it:

```
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=external/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j32
cd build && ./BaseTests && ./DataTests && ./EngineInterfaceTests && ./NetworkTests && ./PersisterTests && ./EngineTests
```

The setup workflow already runs the configure step. The GUI (`./alien`) needs an X11
display and cannot run in a headless environment.

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

Each component lives in `source/<Component>/`. `Interface` holds its API, or the whole
library if the component has no separate implementation. `Impl` holds the implementation
behind the facade. Tests always sit in a `Tests` directory below the `Interface` or `Impl`
directory whose code they test. Includes are relative to `source/`, for example
`#include <Engine/Interface/SimulationFacade.h>`. Only the applications in `source/Apps/`
and the test suites include `*FacadeImpl.h` headers, to register the facade
implementations.

```
source/Apps/Alien/                GUI application, composition root of the facades
source/Apps/Alien-cli/            Command-line interface
source/Base/Interface/            Common utilities, math, logging, Markdown parsing
source/Base/Interface/Tests/      Base unit tests
source/ConsoleUi/Interface/       Console widgets of the CLI and the console mode of the GUI
source/Data/Interface/            Descriptions, genomes, simulation parameters and their services
source/Data/Interface/Tests/      Data unit tests
source/Engine/Impl/               CPU-side engine implementation
source/Engine/Impl/Tests/         CUDA engine integration tests
source/Engine/Interface/          Abstract simulation APIs
source/Engine/Interface/TestData/ Test data shared by the test suites
source/Engine/Interface/Tests/    Engine interface unit tests
source/Engine/Kernels/            CUDA kernels
source/Gui/Impl/                  Dear ImGui GUI, free of graphics API code
source/Gui/Interface/             Abstract GUI API that manages the main window
source/McpTools/Impl/             MCP tools operating on the simulation
source/McpTools/Interface/        Abstract MCP tools API
source/Network/Interface/         HTTP / cloud features, MCP server
source/Network/Interface/Tests/   Network unit tests
source/Persister/Impl/            Worker thread processing the persister requests
source/Persister/Interface/       Abstract persistence APIs, file I/O and serialization
source/Persister/Interface/Tests/ Serialization tests
source/Rendering/Impl/            Vulkan rendering of the simulation and the user interface
source/Rendering/Interface/       Abstract rendering APIs, free of graphics API code
source/Rendering/Shaders/         GLSL shader sources
source/Server/                    Python (FastAPI) server behind the cloud features
external/                         Third-party dependencies incl. the pinned vcpkg submodule
resources/                        Runtime assets
```
