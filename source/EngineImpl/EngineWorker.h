#pragma once

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#include <GL/gl.h>

#include "Base/Definitions.h"

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuConstants.h"
#include "EngineGpuKernels/Definitions.h"

#include "Definitions.h"
#include "DllExport.h"

class EngineWorker
{
public:
    ENGINEIMPL_EXPORT void initCuda();

    ENGINEIMPL_EXPORT void newSimulation(
        IntVector2D worldSize,
        int timestep,
        SimulationParameters const& parameters,
        GpuConstants const& gpuConstants);

    ENGINEIMPL_EXPORT void* registerImageResource(GLuint image);

    ENGINEIMPL_EXPORT void getVectorImage(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        void* const& resource,
        IntVector2D const& imageSize,
        double zoom);

    ENGINEIMPL_EXPORT void updateData(DataChangeDescription const& dataToUpdate);

    ENGINEIMPL_EXPORT void calcNextTimestep();

    ENGINEIMPL_EXPORT void shutdown();

private:
    CudaSimulation _cudaSimulation;

    IntVector2D _worldSize;
    SimulationParameters _parameters;
    GpuConstants _gpuConstants;

    AccessDataTOCache _dataTOCache;
};