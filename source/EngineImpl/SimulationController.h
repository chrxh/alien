#pragma once

#include <thread>

#include "EngineWorker.h"

class ENGINEIMPL_EXPORT SimulationController
{
public:

    void initCuda();

    void newSimulation(
        IntVector2D size,
        int timestep,
        SimulationParameters const& parameters,
        GpuConstants const& gpuConstants);

    void* registerImageResource(GLuint image);

    void getVectorImage(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        void* const& resource,
        IntVector2D const& imageSize,
        double zoom);

    void closeSimulation();

private:
    EngineWorker _worker;
    std::thread* _thread = nullptr;
};
