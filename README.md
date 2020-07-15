alien - Explore digital ecosystems!
===================================
ALiEn is a GPU-based physics engine focused on demands for artificial life computations. It contains a software tool designed to simulate digital organisms embedded in artificial ecosystems and to mimic conditions for (pre-)biotic evolution.

Elementary constituents of simulated matter are free to be programmed in order to extend their capabilities and operate in their environment for special purposes.
Complex and interconnected structures have the potential to perform arbitrary higher-order tasks. Complex structures can emerge spontaneously or by your intervention in the universe.

You have full control of space, time and matter. Explore intriguing worlds which are beyond imagination!

Documentation
=============
Please visit [alien-project.org](https://alien-project.org/documentation.html) for a comprehensive documentation of the program and the underlying model.

How to build
============
You can build alien using Microsoft Visual Studio 2015 or higher. The solution file is contained in msvc/Gui/alien.sln. The following external libaries are necessary:
- Qt 5.8 or higher should be installed
- boost library [version 1.65.1](https://www.boost.org/users/history/version_1_65_1.html) should be installed in external/boost_1_65_1 of the repository
- gtest framework is already contained in the repository in external/gtest
Please note that the (experimental) GPU project in source/ModelGpu/ requiring Nvidia CUDA is not necessary to build alien.

Installer
=========
There is also a Windows installer with 64 bit binaries available at [alien-project.org](https://alien-project.org/download.html).

License
=======
alien is licensed under the [GPLv3](LICENSE).