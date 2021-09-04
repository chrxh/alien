#pragma once

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#include <GL/gl.h>

#include "Base/Definitions.h"

#include "EngineInterface/SimulationParameters.h"
#include "EngineGpuKernels/GpuConstants.h"

#include "DllExport.h"

class CudaSimulation;

class ENGINEIMPL_EXPORT EngineWorker
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

    void shutdown();

private:
    CudaSimulation* _cudaSimulation;
};