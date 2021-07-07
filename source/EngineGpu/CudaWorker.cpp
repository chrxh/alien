#include <functional>

#include <QImage>
#include <QElapsedTimer>
#include <QThread>
#include <QString>
#include <QApplication>
#include <QOpenGLFunctions_3_3_Core>
#include <QOffscreenSurface>

#include "Base/NumberGenerator.h"
#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"
#include "EngineInterface/SpaceProperties.h"
#include "EngineInterface/PhysicalActions.h"
#include "EngineGpuKernels/AccessTOs.cuh"

#include "CudaJobs.h"
#include "CudaWorker.h"
#include "EngineGpuData.h"
#include "DataConverter.h"

CudaWorker::CudaWorker(QObject* parent /*= nullptr*/)
    : QObject(parent)
{
    _surface = new QOffscreenSurface();
    _surface->create();
}

CudaWorker::~CudaWorker()
{
    delete _surface;
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
    _cudaSimulation = new CudaSimulation({ size.x, size.y }, timestep, parameters, cudaConstants);
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
    QSurfaceFormat format;
    format.setMajorVersion(3);
    format.setMinorVersion(3);
    format.setProfile(QSurfaceFormat::CoreProfile);

    _context = new QOpenGLContext();
    _context->setFormat(format);
    _context->create();

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
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();

    std::lock_guard<std::mutex> lock(_mutex);
    if (_jobs.empty()) {
        return;
    }
    bool notify = false;

    for (auto const& job : _jobs) {

        if (auto _job = boost::dynamic_pointer_cast<_GetVectorImageJob>(job)) {
            auto worldRect = _job->getWorldRect();
            auto zoom = _job->getZoom();
            auto resource = _job->getTargetImage();
            auto imageSize = _job->getImageSize();
            auto& mutex = _job->getMutex();

            std::lock_guard<std::mutex> lock(mutex);
            {
                _context->makeCurrent(_surface);

                _cudaSimulation->getVectorImage(
                    {worldRect.p1.x, worldRect.p1.y},
                    {worldRect.p2.x, worldRect.p2.y},
                    resource.data,
                    {imageSize.x, imageSize.y},
                    zoom);
            }
        }

        if (auto _job = boost::dynamic_pointer_cast<_GetDataJob>(job)) {
            auto rect = _job->getRect();
            auto dataTO = _job->getDataTO();
            _cudaSimulation->getSimulationData({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, dataTO);
        }

        if (auto _job = boost::dynamic_pointer_cast<_UpdateDataJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: update data");

            auto rect = _job->getRect();
            auto dataTO = _job->getDataTO();
            _cudaSimulation->getSimulationData({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, dataTO);

            loggingService->logMessage(Priority::Unimportant, "CudaWorker: update data finished 1/3");

            DataConverter converter(dataTO, _numberGenerator, _job->getSimulationParameters(), _cudaSimulation->getCudaConstants());
            converter.updateData(_job->getUpdateDescription());

            loggingService->logMessage(Priority::Unimportant, "CudaWorker: update data finished 2/3");

            _cudaSimulation->setSimulationData({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, dataTO);

            loggingService->logMessage(Priority::Unimportant, "CudaWorker: update data finished 3/3");
        }

        if (auto _job = boost::dynamic_pointer_cast<_SetDataJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: set data");

            auto rect = _job->getRect();
            auto dataTO = _job->getDataTO();
            _cudaSimulation->setSimulationData({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, dataTO);

            loggingService->logMessage(Priority::Unimportant, "CudaWorker: set data finished");
        }

        if (auto _job = boost::dynamic_pointer_cast<_RunSimulationJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: run simulation");
            _simulationRunning = true;
        }

        if (auto _job = boost::dynamic_pointer_cast<_StopSimulationJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: stop simulation");
            _simulationRunning = false;
        }

        if (auto _job = boost::dynamic_pointer_cast<_CalcSingleTimestepJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: calculate single time step");
            _cudaSimulation->calcCudaTimestep();
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: calculate single time step finished");

            Q_EMIT timestepCalculated();
        }

        if (auto _job = boost::dynamic_pointer_cast<_TpsRestrictionJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: restrict time steps per second");
            _tpsRestriction = _job->getTpsRestriction();
        }

        if (auto _job = boost::dynamic_pointer_cast<_SetSimulationParametersJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: set simulation parameters");
            _cudaSimulation->setSimulationParameters(_job->getSimulationParameters());
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: set simulation parameters finished");
        }

        if (auto _job = boost::dynamic_pointer_cast<_SetExecutionParametersJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: set execution parameters");
            _cudaSimulation->setExecutionParameters(_job->getSimulationExecutionParameters());
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: set execution parameters finished");
        }

        if (auto _job = boost::dynamic_pointer_cast<_GetMonitorDataJob>(job)) {
            _job->setMonitorData(_cudaSimulation->getMonitorData());
        }

        if (auto _job = boost::dynamic_pointer_cast<_ClearDataJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: clear data");
            _cudaSimulation->clear();
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: clear data finished");
        }

        if (auto _job = boost::dynamic_pointer_cast<_SelectDataJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: select data");
            auto const pos = _job->getPosition();
            _cudaSimulation->selectData({ pos.x, pos.y });
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: select data finished");
        }

        if (auto _job = boost::dynamic_pointer_cast<_DeselectDataJob>(job)) {
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: deselect data");
            _cudaSimulation->deselectData();
            loggingService->logMessage(Priority::Unimportant, "CudaWorker: deselect data finished");
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

void* CudaWorker::registerImageResource(GLuint image)
{
    std::lock_guard<std::mutex> lock(_mutex);

    QOpenGLFunctions_3_3_Core openGL;
    openGL.initializeOpenGLFunctions();
    openGL.glBindTexture(GL_TEXTURE_2D, image);
    auto result = _cudaSimulation->registerImageResource(image);
    openGL.glBindTexture(GL_TEXTURE_2D, 0);
    return result;
}
