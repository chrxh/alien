#pragma once

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#include <GL/gl.h>

#include "CudaConstants.h"
#include "Definitions.cuh"
#include "DllExport.h"
#include "EngineInterface/ExecutionParameters.h"
#include "EngineInterface/MonitorData.h"

class ENGINEGPUKERNELS_EXPORT CudaSimulation
{
public:
    CudaSimulation(
        int2 const& worldSize,
        int timestep,
        SimulationParameters const& parameters,
        CudaConstants const& cudaConstants);
    ~CudaSimulation();

    void* registerImageResource(GLuint image);

    void calcCudaTimestep();

    void getVectorImage(
        float2 const& rectUpperLeft,
        float2 const& rectLowerRight,
        void* const& resource,
        int2 const& imageSize,
        double zoom);
    void getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
    void setSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);

    void selectData(int2 const& pos);
    void deselectData();

    struct ApplyForceData
    {
        float2 startPos;
        float2 endPos;
        float2 force;
        bool onlyRotation;
    };
    void applyForce(ApplyForceData const& applyData);
    void moveSelection(float2 const& displacement);

    CudaConstants getCudaConstants() const;
    MonitorData getMonitorData();
    int getTimestep() const;
    void setTimestep(int timestep);

    void setSimulationParameters(SimulationParameters const& parameters);
    void setExecutionParameters(ExecutionParameters const& parameters);

    void clear();

private:
    void setCudaConstants(CudaConstants const& cudaConstants);
    void DEBUG_printNumEntries();

private:
    CudaConstants _cudaConstants;
    SimulationData* _cudaSimulationData;
    DataAccessTO* _cudaAccessTO;
    CudaMonitorData* _cudaMonitorData;
};
