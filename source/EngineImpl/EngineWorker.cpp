#include "EngineWorker.h"

#include "EngineGpuKernels/AccessTOs.cuh"
#include "EngineInterface/ChangeDescriptions.h"

#include "AccessDataTOCache.h"
#include "DataConverter.h"

namespace
{
    class CudaAccess
    {
    public:
        CudaAccess(
            std::condition_variable& conditionForAccess,
            std::condition_variable& conditionForWorkerLoop,
            std::atomic<bool>& accessFlag,
            std::atomic<bool> const& isSimulationRunning)
            : _accessFlag(accessFlag)
            , _conditionForWorkerLoop(conditionForWorkerLoop)
        {
            if (!isSimulationRunning.load()) {
                return;
            }
            std::mutex mutex;
            std::unique_lock<std::mutex> uniqueLock(mutex);
            accessFlag.store(true);
            conditionForAccess.wait(uniqueLock);
            conditionForWorkerLoop.notify_all();
        }

        ~CudaAccess()
        {
            _accessFlag.store(false);
            _conditionForWorkerLoop.notify_all();
        }

    private:
        std::atomic<bool>& _accessFlag;
        std::condition_variable& _conditionForWorkerLoop;
    };
}

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

void EngineWorker::registerImageResource(GLuint image)
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning);

    _cudaResource = _cudaSimulation->registerImageResource(image);
}

void EngineWorker::getVectorImage(
    RealVector2D const& rectUpperLeft, 
    RealVector2D const& rectLowerRight, 
    IntVector2D const& imageSize, 
    double zoom)
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning);

    _cudaSimulation->getVectorImage(
        {rectUpperLeft.x, rectUpperLeft.y},
        {rectLowerRight.x, rectLowerRight.y},
        _cudaResource,
        {imageSize.x, imageSize.y},
        zoom);
}

void EngineWorker::updateData(DataChangeDescription const& dataToUpdate)
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning);

    DataAccessTO dataTO = _dataTOCache->getDataTO();
    _cudaSimulation->getSimulationData({0, 0}, int2{_worldSize.x, _worldSize.y}, dataTO);

    DataConverter converter(dataTO, _parameters, _gpuConstants);
    converter.updateData(dataToUpdate);

    _dataTOCache->releaseDataTO(dataTO);

    _cudaSimulation->setSimulationData({0, 0}, int2{_worldSize.x, _worldSize.y}, dataTO);
}

void EngineWorker::calcNextTimestep()
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning);

    _cudaSimulation->calcCudaTimestep();
}

void EngineWorker::beginShutdown()
{
    _isShutdown.store(true);
    _conditionForWorkerLoop.notify_all();
}

void EngineWorker::endShutdown()
{
    _isSimulationRunning = false;
    _isShutdown = false;
    _requireAccess = false;

    _cudaSimulation.reset();
}

void EngineWorker::runThreadLoop()
{
    std::unique_lock<std::mutex> uniqueLock(_mutexForLoop);
    while (true) {
        if (!_isSimulationRunning.load()) {
            
            //sleep...
            _conditionForWorkerLoop.wait(uniqueLock);
        }
        if (_isShutdown.load()) {
            return;
        }
        if(_requireAccess.load()) {
            _conditionForAccess.notify_all();
            _conditionForWorkerLoop.wait(uniqueLock);
        }

        if (_isSimulationRunning.load()) {
            _cudaSimulation->calcCudaTimestep();
        }
    }
}

void EngineWorker::runSimulation()
{
    _isSimulationRunning.store(true);
    _conditionForWorkerLoop.notify_all();
}

void EngineWorker::pauseSimulation()
{
    _isSimulationRunning.store(false);
    _conditionForWorkerLoop.notify_all();
}
