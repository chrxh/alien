#pragma once

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#include <GL/gl.h>

#include "EngineInterface/MonitorData.h"
#include "EngineInterface/GpuConstants.h"

#include "Definitions.cuh"
#include "DllExport.h"

class _CudaSimulation
{
public:
    ENGINEGPUKERNELS_EXPORT static void initCuda();

    ENGINEGPUKERNELS_EXPORT _CudaSimulation(
        int2 const& worldSize,
        int timestep,
        SimulationParameters const& parameters,
        GpuConstants const& cudaConstants);
    ENGINEGPUKERNELS_EXPORT ~_CudaSimulation();

    ENGINEGPUKERNELS_EXPORT void* registerImageResource(GLuint image);

    ENGINEGPUKERNELS_EXPORT void calcCudaTimestep();

    ENGINEGPUKERNELS_EXPORT void getVectorImage(
        float2 const& rectUpperLeft,
        float2 const& rectLowerRight,
        void* cudaResource,
        int2 const& imageSize,
        double zoom);
    ENGINEGPUKERNELS_EXPORT void
    getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
    ENGINEGPUKERNELS_EXPORT void
    setSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);

    ENGINEGPUKERNELS_EXPORT void selectData(int2 const& pos);
    ENGINEGPUKERNELS_EXPORT void deselectData();

    struct ApplyForceData
    {
        float2 startPos;
        float2 endPos;
        float2 force;
        bool onlyRotation;
    };
    ENGINEGPUKERNELS_EXPORT void applyForce(ApplyForceData const& applyData);
    ENGINEGPUKERNELS_EXPORT void moveSelection(float2 const& displacement);

    ENGINEGPUKERNELS_EXPORT GpuConstants getGpuConstants() const;
    ENGINEGPUKERNELS_EXPORT MonitorData getMonitorData();
    ENGINEGPUKERNELS_EXPORT int getTimestep() const;
    ENGINEGPUKERNELS_EXPORT void setTimestep(int timestep);

    ENGINEGPUKERNELS_EXPORT void setSimulationParameters(SimulationParameters const& parameters);

    ENGINEGPUKERNELS_EXPORT void clear();

private:
    void setGpuConstants(GpuConstants const& cudaConstants);

private:
    GpuConstants _gpuConstants;
    SimulationData* _cudaSimulationData;
    DataAccessTO* _cudaAccessTO;
    CudaMonitorData* _cudaMonitorData;
};
