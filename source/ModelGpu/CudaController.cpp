#include <QMessageBox>

#include "CudaController.h"

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"

#include "CudaWorker.h"
#include "CudaJobs.h"
#include "ModelGpuData.h"

namespace
{
	const string ThreadControllerId = "ThreadControllerId";
}

CudaController::CudaController(QObject* parent /*= nullptr*/)
	: QObject(parent)
{
    auto factory = ServiceLocator::getInstance().getService<GlobalFactory>();
    auto numberGenerator = factory->buildRandomNumberGenerator();
    numberGenerator->init(1323781, 2);
    SET_CHILD(_numberGenerator, numberGenerator);

	_worker = new CudaWorker();
	_worker->moveToThread(&_thread);
	connect(_worker, &CudaWorker::timestepCalculated, this, &CudaController::timestepCalculatedWithGpu);
    connect(_worker, &CudaWorker::errorThrown, this, &CudaController::showErrorMessageAndTerminate);
    connect(this, &CudaController::runWorker, _worker, &CudaWorker::run);
	_thread.start();
	_thread.setPriority(QThread::TimeCriticalPriority);
	Q_EMIT runWorker();
}

CudaController::~CudaController()
{
	_worker->terminateWorker();
	_thread.quit();
	if (!_thread.wait(2000)) {
		_thread.terminate();
		_thread.wait();
	}
	delete _worker;
}

void CudaController::init(
    SpaceProperties* space,
    int timestep,
    SimulationParameters const& parameters,
    CudaConstants const& cudaConstants)
{
    _worker->init(space, timestep, parameters, cudaConstants, _numberGenerator);
}

CudaWorker * CudaController::getCudaWorker() const
{
	return _worker;
}

void CudaController::calculate(RunningMode mode)
{
	if (mode == RunningMode::CalcSingleTimestep) {
		CudaJob job = boost::make_shared<_CalcSingleTimestepJob>(ThreadControllerId, false);
		_worker->addJob(job);
	}
	if (mode == RunningMode::OpenEnded) {
		CudaJob job = boost::make_shared<_RunSimulationJob>(ThreadControllerId, false);
		_worker->addJob(job);
	}
	if (mode == RunningMode::DoNothing) {
		CudaJob job = boost::make_shared<_StopSimulationJob>(ThreadControllerId, false);
		_worker->addJob(job);
	}
}

void CudaController::restrictTimestepsPerSecond(optional<int> tps)
{
    auto const job = boost::make_shared<_TpsRestrictionJob>(ThreadControllerId, tps);
	_worker->addJob(job);
}

void CudaController::setSimulationParameters(SimulationParameters const & parameters)
{
    auto const job = boost::make_shared<_SetSimulationParametersJob>(ThreadControllerId, parameters);
	_worker->addJob(job);
}

void CudaController::setExecutionParameters(ExecutionParameters const & parameters)
{
    auto const job = boost::make_shared<_SetExecutionParametersJob>(ThreadControllerId, parameters);
    _worker->addJob(job);
}

void CudaController::timestepCalculatedWithGpu()
{
	Q_EMIT timestepCalculated();
}

void CudaController::showErrorMessageAndTerminate(QString what) const
{
    QMessageBox messageBox;
    messageBox.critical(0, "CUDA error",
        QString("An error has occurred. Please restart the program and check the CUDA parameters.\n\nError message:\n\"%1\"")
        .arg(what));
    exit(EXIT_FAILURE);
}
