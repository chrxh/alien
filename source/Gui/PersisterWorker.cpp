#include "PersisterWorker.h"

#include "SerializationHelperService.h"

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

void PersisterWorker::saveToDisc(std::string const& filename)
{
    {
        std::unique_lock uniqueLock(_jobMutex);
        auto saveToDiscJob = std::make_shared<_SaveToDiscJob>(_idCount++, filename);
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
    auto deserializedData = SerializationHelperService::getDeserializedSerialization(_simController);
    SerializerService::serializeSimulationToFiles(job->getFilename(), deserializedData);
    lock.lock();

    return std::make_shared<_SaveToDiscJobResult>(job->getId());
}
