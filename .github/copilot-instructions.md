## General
ALIEN is an artificial life simulator.

## Repository Structure
- `external/`: External libraries which are not included via vcpkg.
- `resources/`: Resource files which are needed by ALIEN during runtime.
- `scripts/`: PHP script for the server to handle download, upload or info requests.
- `source/`: Contains the C++ and CUDA sources.
- `source/Base/`: General stuff which is mostly not domain related.
- `source/Cli/`: Command line interface for ALIEN.
- `source/EngineGpuKernels/`: Here are the CUDA kernels situated as well as the services for executing them.
- `source/EngineImpl/`: Implements interfaces from `source/EngineInterface/` by invoking services from `source/EngineGpuKernels/`.
- `source/EngineInterface/`: Interfaces for (asynchronous) transfer of data to/from a simulation, calculating time steps and render data.
- `source/EngineTestData/`: Provides a `TestDataFactory` which can be used from test projects.
- `source/EngineTests/`: Contains integration tests for the engine.
- `source/Gui/`: GUI project for ALIEN.
- `source/Network/`: Communication functionalities via https requests.
- `source/NetworkTest/`: Test project for the network project.
- `source/PersisterImpl/`: Implements interfaces from `source/PersisterInterface/` for asynchronous loading, saving, uploading and downloading tasks.
- `source/PersisterInterface/`: Interfaces for the asynchronous transfer of data to/from storage or network.

## GUI
The GUI is based on "Dear ImGui" but customizes the own widgets in `AlienGui`.

## Engine
The GUI and the engine are separated in different projects. The engine is driven by CUDA kernels and is hidden via an interface. From the outside, the engine
can be accessed via `SimulationFacade` from `source/EngineInterface`, whose implementation delegates the requests to `SimulationCudaFacade` from `source/EngineGpuKernels`
and then to specific services that invoke CUDA kernels.

## Main data structures
For the GUI and for serialization purposes the simulation data are stored in description classes which can be found in `Descriptions.h` as well as in `GenomeDescription.h`.
During the data transfer to the GPU they are converted to transfer objects (`ObjectTO.h` and `GenomeTO.h`) and then to objects living only in the GPU memory (`Object.cuh` and `Genome.cuh`).


