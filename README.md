<h1 align="center">
<a href="https://alien-project.org" target="_blank">ALiEn - Explore the world of artificial life</a>
</h1>

<h1 align="center">
<img src="img/alien.png" width=100%>
</h1>
<b>Artificial Life Environment (ALiEn)</b> is an artificial life simulation tool based on a specialized 2D particle engine in CUDA for soft bodies and fluid-like media. Each simulated body consists of a network of <i>smart</i> particles that can be enriched with higher-level functions, ranging from pure information processing capabilities to physical equipment (such as sensors, actuators, weapons, constructors, etc.) whose executions are orchestrated by a signaling system. The bodies can be thought of as agents or digital organisms operating in a common environment.
<br/>
An important goal is to make the simulator user-friendly through a modern user interface and a playful approach. 
<br/><br/>
<b>Note: This branch contains ongoing work for the next major release. For the latest stable version, please visit the <a href="https://github.com/chrxh/alien/tree/master">master branch</a>.</b>

## Main features
### Physics and graphics engine
- Particles for simulating soft body mechanics, heat dissipation, bondings, damages, phase transitions, etc.
- Real-time user interactions with running simulations
- Simulation runs entirely on GPU via CUDA
- Rendering and post-processing via OpenGL using CUDA-OpenGL interoperability

<img src="img/physics engine.gif" width=100%>

### Artificial Life extensions
- Programmable particle actions for simulating digital organisms and studying evolution
- Information and energy transportation layer between connected particles
- Built-in graph editor and scripting environment for designing customized machines and worlds

<img src="img/alife engine.gif" width=100%>

The simulation code is written entirely in CUDA and highly optimized for large-scale real-time simulations of millions of bodies and particles.
The development is driven by the desire to better understand the conditions for (pre-)biotic evolution and the growing complexity of biological systems.

## But for what is this useful?
- A first attempt to answer: Feed your curiosity by watching evolution at work! As soon as self-replicating machines come into play and mutations are turned on, the simulation itself does everything.
- Perhaps the most honest answer: Fun! It is almost like a game with a pretty fast and realistic physics engine. You can make hundreds of thousands of machines accelerate and destroy with the mouse cursor. It feels like playing god in your own universe with your own rules. Different render styles and a visual editor offer fascinating insights into the events. There are a lot of videos on the [YouTube channel](https://youtube.com/channel/UCtotfE3yvG0wwAZ4bDfPGYw) for illustration.
- A more academic answer: A tool to tackle fundamental questions of how complexity or life-like structure may arise from simple components. How do entire ecosystems adapt to environmental changes and find a new equilibrium? How to find conditions that allow open-ended evolution?

**Further information and artworks**
* [Website](https://alien-project.org)
* [YouTube](https://youtube.com/channel/UCtotfE3yvG0wwAZ4bDfPGYw)
* [Twitter](https://twitter.com/chrx_h)
* [Reddit](https://www.reddit.com/r/AlienProject)

## Minimal system requirements
An Nvidia graphics card with compute capability 6.0 or higher is needed. Please check [https://en.wikipedia.org/wiki/CUDA#GPUs_supported](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

## How to build the sources
The build process is mostly automated using the cross-platform CMake build system and the vcpkg package manager, which is included as a Git submodule.

### Getting the sources
To obtain the sources, please open a command prompt in a suitable directory (which should not contain whitespace characters) and enter the following command:
```
git clone --recursive https://github.com/chrxh/alien.git
```
Note: The `--recursive` parameter is necessary to check out the vcpkg submodule as well. Besides that, submodules are not normally updated by the standard `git pull` command. Instead, you need to write `git pull --recurse-submodules`.

### Build instructions
Prerequisites: [CUDA Toolkit 11.2+](https://developer.nvidia.com/cuda-downloads) and a toolchain for CMake (e.g. GCC 9.x+ or [Visual Studio 2019](https://visualstudio.microsoft.com/de/free-developer-offers/))

Build steps:
```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j8
```
If everything goes well, the alien executable can be found under the build directory in `./alien` or `.\Release\alien.exe` depending on the used toolchain and platform.

## Installer
An installer for 64-bit binaries is provided for Windows 10: [download link](https://alien-project.org/media/files/alien-installer-v3.0.0-(preview).zip).
It is updated from the develop branch on a weekly basis. 

In the case that the program crashes for an unknown reason, please refer to the troubleshooting section in [alien-project.org/downloads.html](https://alien-project.org/downloads.html).

## Documentation
Please visit [alien-project.org](https://alien-project.org/documentation/index.html) for a documentation of the program and the underlying model. It contains many brief tutorials that guide the reader into the program in small portions.

## Screenshots
#### Startup screen
<h1 align="center">
<img src="img/screenshot1.png" width=100%>
</h1>

#### Evolving self-replicating machines in action
<h1 align="center">
<img src="img/screenshot2.png" width=100%>
</h1>

#### Explosion inside a large grid of robots
<h1 align="center">
<img src="img/screenshot3.png" width=100%>
</h1>

#### Statistics tools
<h1 align="center">
<img src="img/screenshot4.png" width=100%>
</h1>

#### Visual editor and scripting environment (under development, screenshot is from previous version)
<h1 align="center">
<img src="img/screenshot5.png" width=100%>
</h1>

## License
alien is licensed under the [GPLv3](LICENSE).
