#include "PersisterWorker.h"

#include "EngineInterface/SimulationController.h"

PersisterWorker::PersisterWorker(SimulationController const& simController)
    : _simController(simController)
{
}

void PersisterWorker::runThreadLoop()
{
    try {
        std::unique_lock lock(_jobMutex);
        while (!_isShutdown.load()) {
            _conditionVariable.wait(lock);
            processJobs(lock);
        }
    } catch (std::exception const&) {
        //#TODO
    }
}

void PersisterWorker::shutdown()
{
    _isShutdown = true;
    _conditionVariable.notify_all();
}

void PersisterWorker::saveToDisc(std::string const& filename, float const& zoom, RealVector2D const& center)
{
    {
        std::unique_lock uniqueLock(_jobMutex);
        auto saveToDiscJob = std::make_shared<_SaveToDiscJob>(_idCount++, filename, zoom, center);
        _openJobs.emplace_back(saveToDiscJob);
    }
    _conditionVariable.notify_all();
}

void PersisterWorker::processJobs(std::unique_lock<std::mutex>& lock)
{
    if (_openJobs.empty()) {
        return;
    }

    while (!_openJobs.empty()) {

        auto job = _openJobs.front();
        _openJobs.pop_front();

        if (auto const& saveToDiscJob = std::dynamic_pointer_cast<_SaveToDiscJob>(job)) {
            _inProgressJobs.push_back(job);
            auto jobResult = processSaveToDiscJob(lock, saveToDiscJob);
            _inProgressJobs.pop_back();

            _finishedJobs.emplace_back(jobResult);
        }

    }
}

PersisterJobResult PersisterWorker::processSaveToDiscJob(std::unique_lock<std::mutex>& lock, SaveToDiscJob const& job)
{
    lock.unlock();

    DeserializedSimulation deserializedData;
    deserializedData.auxiliaryData.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
    deserializedData.auxiliaryData.realTime = _simController->getRealTime();
    deserializedData.auxiliaryData.zoom = job->getZoom();
    deserializedData.auxiliaryData.center = job->getCenter();
    deserializedData.auxiliaryData.generalSettings = _simController->getGeneralSettings();
    deserializedData.auxiliaryData.simulationParameters = _simController->getSimulationParameters();
    deserializedData.statistics = _simController->getStatisticsHistory().getCopiedData();
    deserializedData.mainData = _simController->getClusteredSimulationData();

    SerializerService::serializeSimulationToFiles(job->getFilename(), deserializedData);
    lock.lock();

    return std::make_shared<_SaveToDiscJobResult>(job->getId());
}
