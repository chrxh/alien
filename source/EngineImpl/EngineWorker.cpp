#include "EngineWorker.h"

#include "EngineGpuKernels/AccessTOs.cuh"

void EngineWorker::initCuda()
{
    CudaSimulation::initCuda();
}

void EngineWorker::newSimulation(
    IntVector2D size,
    int timestep,
    SimulationParameters const& parameters,
    GpuConstants const& gpuConstants)
{
    _cudaSimulation = new CudaSimulation({size.x, size.y}, timestep, parameters, gpuConstants);
}

void* EngineWorker::registerImageResource(GLuint image)
{
    return _cudaSimulation->registerImageResource(image);
}

void EngineWorker::getVectorImage(
    RealVector2D const& rectUpperLeft, 
    RealVector2D const& rectLowerRight, 
    void* const& resource, 
    IntVector2D const& imageSize, 
    double zoom)
{
    _cudaSimulation->getVectorImage(
        {rectUpperLeft.x, rectUpperLeft.y},
        {rectLowerRight.x, rectLowerRight.y},
        resource,
        {imageSize.x, imageSize.y},
        zoom);
}

void EngineWorker::shutdown()
{
    delete _cudaSimulation;
}
