#include "CudaWorker.h"
#include "CudaJob.h"

#include "CudaController.h"

namespace
{
	const string ThreadControllerId = "ThreadControllerId";
}

CudaController::CudaController(QObject* parent /*= nullptr*/)
	: QObject(parent)
{
	_worker = new CudaWorker();
	_worker->moveToThread(&_thread);
	connect(_worker, &CudaWorker::timestepCalculated, this, &CudaController::timestepCalculatedWithGpu);
	connect(this, &CudaController::runWorker, _worker, &CudaWorker::run);
	_thread.start();
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

void CudaController::init(SpaceProperties *space, SimulationParameters const& parameters)
{
	_worker->init(space, parameters);
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
	if (mode == RunningMode::OpenEndedSimulation) {
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
	CudaJob job = boost::make_shared<_TpsRestrictionJob>(ThreadControllerId, tps);
	_worker->addJob(job);
}

void CudaController::setSimulationParameters(SimulationParameters const & parameters)
{
	CudaJob job = boost::make_shared<_SetSimulationParametersJob>(ThreadControllerId, parameters);
	_worker->addJob(job);
}

void CudaController::timestepCalculatedWithGpu()
{
	Q_EMIT timestepCalculated();
}

