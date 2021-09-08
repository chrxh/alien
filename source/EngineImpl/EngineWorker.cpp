#include "EngineWorker.h"

#include "EngineGpuKernels/AccessTOs.cuh"
#include "EngineInterface/ChangeDescriptions.h"

#include "AccessDataTOCache.h"
#include "DataConverter.h"

void EngineWorker::initCuda()
{
    _CudaSimulation::initCuda();
}

void EngineWorker::newSimulation(
    IntVector2D size,
    int timestep,
    SimulationParameters const& parameters,
    GpuConstants const& gpuConstants)
{
    _worldSize = size;
    _parameters = parameters;
    _gpuConstants = gpuConstants;
    _dataTOCache = boost::make_shared<_AccessDataTOCache>(gpuConstants);
    _cudaSimulation = boost::make_shared<_CudaSimulation>(int2{size.x, size.y}, timestep, parameters, gpuConstants);
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

void EngineWorker::updateData(DataChangeDescription const& dataToUpdate)
{
    DataAccessTO dataTO = _dataTOCache->getDataTO();
    _cudaSimulation->getSimulationData({0, 0}, int2{_worldSize.x, _worldSize.y}, dataTO);

    DataConverter converter(dataTO, _parameters, _gpuConstants);
    converter.updateData(dataToUpdate);

    _cudaSimulation->setSimulationData({0, 0}, int2{_worldSize.x, _worldSize.y}, dataTO);
}

ENGINEIMPL_EXPORT void EngineWorker::calcNextTimestep()
{
    _cudaSimulation->calcCudaTimestep();
}

void EngineWorker::shutdown()
{
    _cudaSimulation.reset();
}
