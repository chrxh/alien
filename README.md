<h1 align="center">
<a href="https://alien-project.org" target="_blank">ALiEn - Explore the world of artificial life</a>
</h1>

<h1 align="center">
<img src="img/alien.png" width=100%>
</h1>
Artificial Life Environment (ALiEn) is a simulation program based on a specialized 2D physics and rendering engine in CUDA. Each simulated body has a graph-like structure of connected building blocks that can either be programmed or equipped with devices to act physically (accelerators, sensors, weapons, constructors, etc.). These internal body processes are triggered by signals coming from circulating tokens. Therefore, it is more appropriate to think of them as small machines or agents.

## Main features
- realistic physical calculations of heat dissipation, collisions, bondings, damages, rotational forces, etc.
- programmable matter approach for simulating digital organisms and evolution
- built-in code and graph editor for designing own machines 
- simulation and rendering on GPU

The simulation code is written entirely in CUDA and highly optimized for large-scale real-time simulations of millions of bodies and particles.
The development is driven by the desire to better understand the conditions for (pre-)biotic evolution and the growing complexity of biological systems.

## But for what is this useful?
- A first attempt to answer:  Watch evolution at work! As soon as self-replicating machines come into play and mutations are turned on, the simulation itself does everything.
- Second best answer is: Fun! It is almost like a game with a pretty fast and realistic physics engine. You can make hundreds of thousands of machines accelerate and destroy with the mouse cursor. It feels like playing god in your own universe with your own rules. Different render styles offer a fascinating visual insight into the events. There are a lot of videos on the YouTube channel for illustration.
- A more academic answer: A tool to tackle fundamental questions of how complexity or life-like structure may arise from simple components. How do entire ecosystems adapt to environmental changes and find a new equilibrium? How to find conditions that allow open-ended evolution?

<img src="img/engine.gif" width=100%>

**Further information and artworks**
* [Website](https://alien-project.org)
* [YouTube](https://youtube.com/channel/UCtotfE3yvG0wwAZ4bDfPGYw)
* [Twitter](https://twitter.com/chrx_h)

## Installer
An installer for 64-bit binaries is provided for Windows 10: [download link](https://alien-project.org/downloads.html).

## Documentation
Please visit [alien-project.org](https://alien-project.org/documentation.html) for a comprehensive documentation of the program and the underlying model.

## Screenshots
#### Startup screen
<h1 align="center">
<img src="img/screenshot1.png" width=100%>
</h1>

#### Evolving replicating machines in action
<h1 align="center">
<img src="img/screenshot3.png" width=100%>
</h1>

#### Debris after heavy impact
<h1 align="center">
<img src="img/screenshot5.png" width=100%>
</h1>

#### Graph structure of the bodies
<h1 align="center">
<img src="img/screenshot7.png" width=100%>
</h1>

#### Visual editor for programming the machines
<h1 align="center">
<img src="img/screenshot8.png" width=100%>
</h1>

## How to build the sources
To build alien you need Microsoft Visual Studio 2019. You find the solution file in msvc/alien/alien.sln.
The following third-party libaries are necessary and should be installed:
- [Qt 6.0.1](https://www.qt.io/download)
- [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)
- [boost library version 1.75.0](https://www.boost.org/users/history/version_1_75_0.html) needs to be installed in external/boost_1_75_0

## License
alien is licensed under the [GPLv3](LICENSE).

<p align="left"> <img src="https://komarev.com/ghpvc/?username=chrxh&label=Page%20views&color=0e75b6&style=flat" alt="chrxh" /> </p>
