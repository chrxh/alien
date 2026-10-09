<h1 align="center">
<a href="https://alien-project.org" target="_blank">ALIEN: Explore worlds of artificial life</a>
</h1>

<p align="center">
A GPU-accelerated artificial life simulator for studying emergent ecosystems<br>
and the conditions for open-ended evolution
</p>

<p align="center">
🏆 <b>First prize at the <a href="https://youtu.be/qwbMGPkoJmg" target="_blank">ALIFE 2024 Virtual Creatures Competition</a></b><br>
<sub>with the video <i>Emerging Ecosystems</i></sub>
</p>

<p align="center">
<a href="#windows-nightly-build"><img src="https://img.shields.io/badge/download-nightly%20build-59DACE?style=flat-square" alt="Download nightly build"></a>
<a href="https://hub.docker.com/r/chrxh/alien" target="_blank"><img src="https://img.shields.io/badge/docker-chrxh%2Falien-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker image"></a>
<a href="resources/docs/getting-started.md"><img src="https://img.shields.io/badge/docs-built--in-8A63D2?style=flat-square" alt="Documentation"></a>
<a href="https://discord.gg/7bjyZdXXQ2" target="_blank"><img src="https://img.shields.io/badge/discord-join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD%203--Clause-blue?style=flat-square" alt="License: BSD 3-Clause"></a>
</p>

![Preview](https://github.com/user-attachments/assets/ee578848-7dd7-458d-873f-89662a7c15f0)

## 🧬 About

**ALIEN** (**A**rtificial **LI**fe **EN**vironment) is an artificial life simulator built on a specialized 2D particle engine for soft bodies and fluids. Every object in a simulated world consists of particles. They can be bonded into elastic solids, flow as fluids or act as **cells** with specialized functions such as sensors, muscles, weapons and constructors. Each cell is controlled by its own small neural network. Networks of cells form **creatures**, digital organisms that perceive their environment, move, feed, compete and build offspring according to their **genome**. Once mutations are switched on, the simulation needs no further intervention. Populations can adapt, lineages diverge and entire ecosystems emerge.

The engine runs entirely on the graphics card and simulates worlds with millions of particles in real time. You can interact with a running world at any moment: draw matter, push creatures around, change the laws of physics or let an AI agent do it for you.

The development is driven by the desire to better understand the conditions for (pre-)biotic evolution, the growing complexity of biological systems and the open question of how evolution can keep producing novelty. At the same time, ALIEN aims to be approachable, with a modern user interface, appealing rendering and a playful spirit.

## ⚡ Highlights

### Physics on the GPU
- Particle-based simulation of soft and rigid bodies, fluids, adhesion, fracture and damage
- Millions of particles in real time, written entirely in CUDA and also available for AMD GPUs
- Rendering and post-processing via Vulkan with CUDA-Vulkan interoperability
- Real-time interaction with the running simulation

https://github.com/user-attachments/assets/fad1b719-5662-4840-aa75-165346d62cdd

### Digital organisms
- Creatures are networks of cells, held together by elastic connections that carry energy and signals
- 15 cell types, among them sensors, muscles, attackers, injectors, digestors, memory cells and communicators
- Every cell runs a small neural network. Together, these networks form the nervous system of a creature.
- Genomes are blueprints made of genes. Cells read them and build offspring cell by cell.
- Energy never appears out of nothing. It circulates between cells, free energy particles and an optional external pool.

### Evolution without a fitness function
- Mutations of neural networks, cell properties and the genome structure, with mutation rates that can evolve themselves (meta-mutations)
- Selection arises naturally from limited energy, predation and competition for space
- Layers and radiation sources let the simulation parameters vary across the world and create diverse habitats
- Evolution dashboard with lineages, population statistics and long-term plots
- Autosave with save points and flashbacks to return to interesting moments

https://user-images.githubusercontent.com/73127001/229569056-0db6562b-0147-43c8-a977-5f12c1b6277b.mp4

### AI agents
- Built-in MCP server: connect an AI agent and describe in plain language what you want
- Agents build worlds, design genomes, run experiments, read statistics and inspect the result through screenshots

### Tools for creators
- Genome editor with live preview for designing your own creatures
- Freehand and geometric drawing tools and an image converter that turns pictures into matter
- Inspection windows for every object, creature and genome, plus mass operations
- Built-in documentation with tutorials and a complete reference (press F1)

### Simulation browser
- Built-in access to the simulations and genomes on the ALIEN server, no account needed for browsing and downloading
- **Featured** worlds of the ALIEN project and **Community** worlds shared by other users, shown as a gallery with preview pictures or as a table with folders
- With an account, upload your own simulations and genomes and keep them private or share them with the community
- React to the work of others and sort the gallery by reactions, date or downloads

## 🤖 Let an AI agent drive ALIEN

ALIEN contains a built-in server for the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). Connect Claude Code or any other AI agent that supports MCP and simply describe what you want, for example:

- *"Create a mysterious deep sea scene with glowing jellyfish-like creatures."*
- *"Design a small creature that swims towards energy particles and place five of them."*
- *"Set up an evolution experiment with simple self-replicators and report how the population develops."*
- *"My population dies out after a while. Check the energy balance and suggest better parameters."*

The chapter [AI agents](resources/docs/ai-agents.md) of the documentation explains the setup in detail and contains many more example prompts.

## ❓ But what is this useful for?

- **The curious answer:** Watching evolution at work. Once self-reproducing creatures and mutations come into play, the simulation does the rest.
- **The honest answer:** Fun! A fast physics engine lets you push, smash and rebuild hundreds of thousands of objects with the mouse. It feels like playing god in a universe with your own rules.
- **The academic answer:** Artificial life research. How does complexity arise from simple components? How do ecosystems adapt to environmental change? And which conditions keep evolution open-ended instead of letting it stagnate?
- **The artistic answer:** Generative art. Evolution is a creative force that leads to ever new forms and behaviors.

Plenty of examples can be found on the [YouTube channel](https://youtube.com/channel/UCtotfE3yvG0wwAZ4bDfPGYw).

## 🌌 Gallery

**Plant-like populations around a radiation source**

![Plant-like populations around a radiation source](https://user-images.githubusercontent.com/73127001/229311601-839649a6-c60c-4723-99b3-26086e3e4340.jpg)

**Close-up of different organisms showing their cell networks**

![Close-up of different organisms showing their cell networks](https://user-images.githubusercontent.com/73127001/229311604-3ee433d4-7dd8-46e2-b3e6-489eaffbda7b.jpg)

**Swarms attacking an ecosystem**

![Swarms attacking an ecosystem](https://user-images.githubusercontent.com/73127001/229311606-2f590bfb-71a8-4f71-8ff7-7013de9d7496.jpg)

**Genome editor**

<img width="1920" height="1080" alt="Genome editor" src="https://github.com/user-attachments/assets/0b5c8328-31c7-46bb-816c-d4c0e0315f50" />

## 💻 Installation

### Supported platforms

|                    | NVIDIA GPU (CUDA)                   | AMD GPU (HIP or SCALE)              |
| ------------------ | ----------------------------------- | ----------------------------------- |
| **Windows**        | nightly build or build from sources | nightly build or build from sources |
| **Linux**          | build from sources                  | build from sources                  |
| **Cloud instance** | Docker image, headless              | not available                       |

NVIDIA: compute capability 7.5 or higher, i.e. GeForce RTX 20 series or newer ([list of supported GPUs](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)). AMD: RDNA2 or newer.

### Windows: nightly build

Download and unpack https://alien-project.org/files/alien-develop.zip, which is built every night from the `develop` branch. It contains two executables:
- `alien.exe` for NVIDIA GPUs
- `alien-amd.exe` for AMD GPUs (RDNA2, RDNA3 and RDNA4)

Start it directly from the unpacked folder, otherwise it will not find the resource folder. If the program crashes for an unknown reason, please refer to the [troubleshooting](#-troubleshooting) section below.

### Cloud and Docker

For long runs without local hardware, a nightly image containing the headless [command-line interface](#command-line-interface) is published on Docker Hub as `chrxh/alien:nightly`. It holds `cli` and its resource folder. There is no GUI in the image.

On a rented GPU instance (for example [vast.ai](https://vast.ai)), enter `chrxh/alien:nightly` as the instance image and filter the offers for a GeForce RTX 20 to 50 series GPU. Upload your simulation file to the instance with `scp`, then connect via SSH and start it by hand:
```
cd /opt/alien
cli -i example.sim -o output.sim
```

Locally, with an NVIDIA GPU and the NVIDIA container toolkit installed:
```
docker run --rm --gpus all -v "$PWD":/data --entrypoint cli chrxh/alien:nightly -i /data/example.sim -o /data/output.sim -t 1000
```

### Building from the sources

Windows and Linux are built the same way, using the cross-platform CMake build system, the **Ninja** build tool and the vcpkg package manager, which is included as a Git submodule.

**Getting the sources**

Open a command prompt in a suitable directory (which should not contain whitespace characters) and enter:
```
git clone --recursive https://github.com/chrxh/alien.git
```
The `--recursive` parameter is necessary to check out the vcpkg submodule as well. Submodules are not updated by a plain `git pull`, so use `git pull --recurse-submodules` instead.

**Prerequisites**
- [CUDA Toolkit 11.2+](https://developer.nvidia.com/cuda-downloads) for NVIDIA GPUs, or [ROCm 7.2+](https://rocm.docs.amd.com/) providing HIP for AMD GPUs
- Windows: [Visual Studio](https://visualstudio.microsoft.com/vs/) with the "Desktop development with C++" and "C++ CMake tools for Windows" components (the latter ships Ninja). The MSVC environment must be active while building.
- Linux: GCC, ninja-build, pkg-config and the X11 development libraries:
  ```
  sudo apt-get install ninja-build libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libxext-dev libxfixes-dev libgl1-mesa-dev libglu-dev libxcb1-dev pkg-config
  ```

**NVIDIA GPUs (CUDA)**

On Windows, just run `build-windows-ninja.bat` from the repository root. It sets up the MSVC environment and locates Ninja and CMake from your Visual Studio installation automatically.

On Linux, or on Windows from a *Developer Command Prompt*, invoke the CMake preset directly:
```
cmake --preset ninja
cmake --build --preset ninja-release
```
The executable is written to `build-ninja/Release/` (`alien.exe` on Windows, `alien` on Linux) and has to be started from there, otherwise it will not find the resource folder.

**AMD GPUs (HIP)**

The HIP path builds the same sources for AMD GPUs in place of CUDA. On Windows: `build-windows-ninja.bat HIP`, which writes to `build-ninja-hip/Release/` and covers RDNA2, RDNA3 and RDNA4.

On Linux:
```
cmake --preset ninja-hip -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build --preset ninja-hip-release
```
`CMAKE_HIP_ARCHITECTURES` selects the target architecture (`gfx1100` for RDNA3, `gfx90a` for CDNA2 / MI200). If omitted, it is auto-detected from the GPUs of the host. `CMAKE_PREFIX_PATH` lets `find_package(hip)` locate the ROCm installation when CMake uses the vcpkg toolchain. Adjust it if ROCm is installed elsewhere.

**AMD GPUs (SCALE)**

Alternatively, the unchanged CUDA sources can be compiled for AMD GPUs with [SCALE](https://docs.scale-lang.com) (Linux), whose `nvcc` is a drop-in replacement for the NVIDIA one:
```
cmake --preset ninja -DCMAKE_CUDA_COMPILER=/opt/scale/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=gfx1100
cmake --build --preset ninja-release
```

### Command-line interface

Besides the graphical program, ALIEN includes `cli` (`cli.exe` on Windows), which runs simulations without a window. It is useful for long experiments, performance measurements and the automated evaluation of simulations with different parameters. For example,
```
cli -i example.sim -o output.sim -t 1000
```
runs the simulation file `example.sim` for 1000 time steps and writes the result to `output.sim`. Without `-t`, the simulation runs until you press Q or Ctrl+C. All options are described in the chapter [Files and command line](resources/docs/files.md#command-line-interface).

## 📘 Documentation

ALIEN comes with a complete documentation built into the program. Press **F1** to open it. It is organized in three parts:

- **Start here:** [Getting started](resources/docs/getting-started.md), [the user interface](resources/docs/user-interface.md) and [AI agents](resources/docs/ai-agents.md)
- **Explore:** the central concepts step by step, from the [physics sandbox](resources/docs/sandbox.md) and [how life works](resources/docs/how-life-works.md) to [your first creature](resources/docs/first-creature.md), [evolution experiments](resources/docs/evolution.md) and [generative art](resources/docs/generative-art.md)
- **Reference:** [cell types](resources/docs/cell-types.md), [genomes](resources/docs/genomes.md), [neural networks](resources/docs/neural-networks.md), [energy](resources/docs/energy.md), [simulation parameters](resources/docs/simulation-parameters.md) and more

The same chapters can also be read here on GitHub in [resources/docs](resources/docs).

## 🔎 Troubleshooting

If ALIEN does not start or crashes, please make sure that:
1) You have a supported graphics card: an NVIDIA GPU with compute capability 7.5 or higher (for example GeForce RTX 20 series) or an AMD GPU of the RDNA2 generation or newer.
2) You have the latest graphics driver installed. It has to support Vulkan 1.3.
3) The name of the installation directory (including the parent directories) contains no non-English characters. On Windows, the user name should not contain such characters either.
4) ALIEN has write access to its own directory.
5) With multiple graphics cards, your primary monitor is connected to the card that ALIEN uses. ALIEN computes and renders on the same card and chooses the one with the highest compute capability.
6) On computers with integrated and dedicated graphics, ALIEN runs on the dedicated card. On Windows, open the *Graphics settings*, add `alien.exe`, click *Options* and choose *High performance*.

If the error still occurs, enable **Settings > Debug mode**, reproduce the error and create a [GitHub issue](https://github.com/chrxh/alien/issues) with the `log.txt` attached. More solutions can be found in the chapter [Troubleshooting](resources/docs/troubleshooting.md).

## 💬 Community

- [Discord](https://discord.gg/7bjyZdXXQ2) for discussions, new developments and feedback around ALIEN and artificial life in general
- [YouTube](https://youtube.com/channel/UCtotfE3yvG0wwAZ4bDfPGYw)
- [Reddit](https://www.reddit.com/r/AlienProject)
- [X (Twitter)](https://twitter.com/chrx_h)
- [Website](https://alien-project.org)

## 📖 Citing ALIEN

If you use ALIEN in a scientific publication, please cite:

> Heinemann, C.: *Artificial Life Environment*. Informatik-Spektrum **31**, 55–61 (2008). https://doi.org/10.1007/s00287-007-0205-1

```bibtex
@article{Heinemann2008,
  author  = {Heinemann, Christian},
  title   = {Artificial Life Environment},
  journal = {Informatik-Spektrum},
  volume  = {31},
  number  = {1},
  pages   = {55--61},
  year    = {2008},
  doi     = {10.1007/s00287-007-0205-1}
}
```

Note that this paper describes the beginnings of what has become ALIEN. The current simulator differs substantially from the system presented there. You are therefore equally welcome to cite this repository instead of the paper or alongside it: https://github.com/chrxh/alien

If you adopt ideas or concepts from ALIEN without using the software itself, that is fine but please just mention ALIEN and link to this repository.

## 🧩 Contributing

Contributions to the project are very welcome. The most convenient way is to communicate via [GitHub Issues](https://github.com/chrxh/alien/issues), [Pull requests](https://github.com/chrxh/alien/pulls) or the [Discussion forum](https://github.com/chrxh/alien/discussions) depending on the subject. For example, it could be
- Providing new content (simulation or genome files)
- Producing or sharing media files
- Reporting of bugs, wanted features, questions or feedback via GitHub Issues or in the Discussion forum
- Pull requests for bug fixes, code cleanings, optimizations or minor tweaks. If you want to implement new features, refactorings or other major changes, please use the [Discussion forum](https://github.com/chrxh/alien/discussions) for consultation and coordination in advance.
- Extensions or corrections of the documentation, which is written in Markdown in [resources/docs](resources/docs)
## 💎 Credits and dependencies

ALIEN has been initiated, mainly developed and maintained by [Christian Heinemann](mailto:heinemann.christian@gmail.com). Many thanks to everyone who has contributed to this project in any way. In alphabetical order:
- [dguerizec](https://github.com/dguerizec)
- [jeffdaily](https://github.com/jeffdaily)
- [Enitoni](https://github.com/Enitoni) via Discord
- [Gardene-el](https://github.com/Gardene-el)
- [mpersano](https://github.com/mpersano)
- [TheBarret](https://github.com/TheBarret) via Discord
- tilsk via Discord
- [tlemo](https://github.com/tlemo)
- [Will Allen](https://github.com/willjallen)

The following external libraries are used:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Dear ImGui](https://github.com/ocornut/imgui)
- [ImPlot](https://github.com/epezent/implot)
- [ImFileDialog](https://github.com/dfranx/ImFileDialog)
- [boost](https://www.boost.org)
- [GLFW](https://www.glfw.org)
- [glslang](https://github.com/KhronosGroup/glslang)
- [volk](https://github.com/zeux/volk)
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers)
- [stb](https://github.com/nothings/stb)
- [cereal](https://github.com/USCiLab/cereal)
- [Zstandard](https://github.com/facebook/zstd)
- [OpenSSL](https://github.com/openssl/openssl)
- [cpp-httplib](https://github.com/yhirose/cpp-httplib)
- [googletest](https://github.com/google/googletest)
- [vcpkg](https://vcpkg.io/en/index.html)
- [WinReg](https://github.com/GiovanniDicanio/WinReg)
- [CLI11](https://github.com/CLIUtils/CLI11)
- [MD4C](https://github.com/mity/md4c)

Free icons and icon font:
  - [IconFontCppHeaders](https://github.com/juliettef/IconFontCppHeaders)
  - [Iconduck](https://iconduck.com) (Noto Emoji by Google, [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt))
  - [Iconfinder](https://www.iconfinder.com) (Bogdan Rosu Creative, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0))
  - [People icons created by Freepik - Flaticon](https://www.flaticon.com/free-icons/people) ([Flaticon license](https://media.flaticon.com/license/license.pdf))

## 🧾 License

ALIEN is licensed under the [BSD 3-Clause](LICENSE) license.

<a href="https://hellogithub.com/repository/d53e3c352f294f72a1bfd8f48ac0f866" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=d53e3c352f294f72a1bfd8f48ac0f866&claim_uid=dKUYgLps8t45BW7&theme=small" alt="Featured｜HelloGitHub" /></a>
