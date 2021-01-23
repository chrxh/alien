#include <functional>

#include <QImage>
#include <QElapsedTimer>
#include <QThread>
#include <QString>

#include "Base/NumberGenerator.h"
#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/PhysicalActions.h"

#include "AccessTOs.cuh"
#include "CudaJobs.h"
#include "CudaWorker.h"
#include "ModelGpuData.h"
#include "DataConverter.h"

CudaWorker::~CudaWorker()
{
	delete _cudaSimulation;
}

void CudaWorker::init(
    SpaceProperties* space,
    int timestep,
    SimulationParameters const& parameters,
    CudaConstants const& cudaConstants,
    NumberGenerator* numberGenerator)
{
    _numberGenerator = numberGenerator;

    auto size = space->getSize();
	delete _cudaSimulation;
    try {
    	_cudaSimulation = new CudaSimulation({ size.x, size.y }, timestep, parameters, cudaConstants);
    }
    catch (std::exception const& exception) {
        terminateWorker();
        Q_EMIT errorThrown(exception.what());
    }
}

void CudaWorker::terminateWorker()
{
	std::lock_guard<std::mutex> lock(_mutex);
	_terminate = true;
	_condition.notify_all();
}

void CudaWorker::addJob(CudaJob const & job)
{
	std::lock_guard<std::mutex> lock(_mutex);
	_jobs.push_back(job);
	_condition.notify_all();
}

vector<CudaJob> CudaWorker::getFinishedJobs(string const & originId)
{
	std::lock_guard<std::mutex> lock(_mutex);
	vector<CudaJob> result;
	vector<CudaJob> remainingJobs;
	for (auto const& job : _finishedJobs) {
		if (job->getOriginId() == originId) {
			result.push_back(job);
		}
		else {
			remainingJobs.push_back(job);
		}
	}
	_finishedJobs = remainingJobs;
	return result;
}

void CudaWorker::run()
{
    try {
        do {
		    QElapsedTimer timer;
		    timer.start();

            processJobs();

            if (isSimulationRunning()) {
                _cudaSimulation->calcCudaTimestep();

                if (_tpsRestriction) {
                    int remainingTime = 1000000 / (*_tpsRestriction) - timer.nsecsElapsed() / 1000;
                    if (remainingTime > 0) {
                        QThread::usleep(remainingTime);
                    }
                }
                Q_EMIT timestepCalculated();
            }

		    std::unique_lock<std::mutex> uniqueLock(_mutex);
		    if (!_jobs.empty() && !_terminate) {
			    _condition.wait(uniqueLock, [this]() {
				    return !_jobs.empty() || _terminate;
			    });
		    }
	    } while (!isTerminate());
    }
    catch (std::exception const& exeception)
    {
        terminateWorker();
        Q_EMIT errorThrown(exeception.what());
    }
}

void CudaWorker::processJobs()
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (_jobs.empty()) {
        return;
    }
    bool notify = false;

    for (auto const& job : _jobs) {

        if (auto _job = boost::dynamic_pointer_cast<_GetImageJob>(job)) {
            auto rect = _job->getRect();
            auto image = _job->getTargetImage();
            auto& mutex = _job->getMutex();

            std::lock_guard<std::mutex> lock(mutex);
            _cudaSimulation->getSimulationImage({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, image->bits());
        }

        if (auto _job = boost::dynamic_pointer_cast<_GetDataJob>(job)) {
            auto rect = _job->getRect();
            auto dataTO = _job->getDataTO();
            _cudaSimulation->getSimulationData({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, dataTO);
        }

        if (auto _job = boost::dynamic_pointer_cast<_UpdateDataJob>(job)) {
            auto rect = _job->getRect();
            auto dataTO = _job->getDataTO();
            _cudaSimulation->getSimulationData({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, dataTO);

            DataConverter converter(dataTO, _numberGenerator, _job->getSimulationParameters(), _cudaSimulation->getCudaConstants());
            converter.updateData(_job->getUpdateDescription());

            _cudaSimulation->setSimulationData({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, dataTO);

        }

        if (auto _job = boost::dynamic_pointer_cast<_SetDataJob>(job)) {
            auto rect = _job->getRect();
            auto dataTO = _job->getDataTO();
            _cudaSimulation->setSimulationData({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, dataTO);
        }

        if (auto _job = boost::dynamic_pointer_cast<_RunSimulationJob>(job)) {
            _simulationRunning = true;
        }

        if (auto _job = boost::dynamic_pointer_cast<_StopSimulationJob>(job)) {
            _simulationRunning = false;
        }

        if (auto _job = boost::dynamic_pointer_cast<_CalcSingleTimestepJob>(job)) {
            _cudaSimulation->calcCudaTimestep();
            Q_EMIT timestepCalculated();
        }

        if (auto _job = boost::dynamic_pointer_cast<_TpsRestrictionJob>(job)) {
            _tpsRestriction = _job->getTpsRestriction();
        }

        if (auto _job = boost::dynamic_pointer_cast<_SetSimulationParametersJob>(job)) {
            _cudaSimulation->setSimulationParameters(_job->getSimulationParameters());
        }

        if (auto _job = boost::dynamic_pointer_cast<_SetExecutionParametersJob>(job)) {
            _cudaSimulation->setExecutionParameters(_job->getSimulationExecutionParameters());
        }

        if (auto _job = boost::dynamic_pointer_cast<_GetMonitorDataJob>(job)) {
            _job->setMonitorData(_cudaSimulation->getMonitorData());
        }

        if (auto _job = boost::dynamic_pointer_cast<_ClearDataJob>(job)) {
            _cudaSimulation->clear();
        }

        if (auto _job = boost::dynamic_pointer_cast<_SelectDataJob>(job)) {
            auto const pos = _job->getPosition();
            _cudaSimulation->selectData({ pos.x, pos.y });
        }

        if (auto _job = boost::dynamic_pointer_cast<_DeselectDataJob>(job)) {
            _cudaSimulation->deselectData();
        }

        if (auto _job = boost::dynamic_pointer_cast<_PhysicalActionJob>(job)) {
            auto action = _job->getAction();
            if (auto _action = boost::dynamic_pointer_cast<_ApplyForceAction>(action)) {
                float2 startPos = { _action->getStartPos().x(), _action->getStartPos().y() };
                float2 endPos = { _action->getEndPos().x(), _action->getEndPos().y() };
                float2 force = { _action->getForce().x(), _action->getForce().y() };
                _cudaSimulation->applyForce({ startPos, endPos, force, false });
            }
            if (auto _action = boost::dynamic_pointer_cast<_ApplyRotationAction>(action)) {
                float2 startPos = { _action->getStartPos().x(), _action->getStartPos().y() };
                float2 endPos = { _action->getEndPos().x(), _action->getEndPos().y() };
                float2 force = { _action->getForce().x(), _action->getForce().y() };
                _cudaSimulation->applyForce({ startPos, endPos, force, true });
            }
            if (auto _action = boost::dynamic_pointer_cast<_MoveSelectionAction>(action)) {
                float2 displacement = { _action->getDisplacement().x(), _action->getDisplacement().y() };
                _cudaSimulation->moveSelection(displacement);
            }
        }

        if (job->isNotifyFinish()) {
            notify = true;
        }
    }
    if (notify) {
        _finishedJobs.insert(_finishedJobs.end(), _jobs.begin(), _jobs.end());
        _jobs.clear();
        Q_EMIT jobsFinished();
    }
    else {
        _jobs.clear();
    }
}

bool CudaWorker::isTerminate()
{
	std::lock_guard<std::mutex> lock(_mutex);
	return _terminate;
}

bool CudaWorker::isSimulationRunning()
{
	std::lock_guard<std::mutex> lock(_mutex);
	return _simulationRunning;
}

int CudaWorker::getTimestep()
{
    if (isTerminate()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    return _cudaSimulation->getTimestep();
}

void CudaWorker::setTimestep(int timestep)
{
    if (isTerminate()) {
        return;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    return _cudaSimulation->setTimestep(timestep);
}
