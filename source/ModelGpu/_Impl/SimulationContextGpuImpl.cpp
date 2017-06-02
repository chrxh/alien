#include "Model/SpaceMetricApi.h"
#include "Model/Metadata/SymbolTable.h"
#include "Model/Context/SimulationParameters.h"
#include "Model/SpaceMetricApi.h"

#include "GpuWorker.h"
#include "GpuObserver.h"
#include "SimulationContextGpuImpl.h"

SimulationContextGpuImpl::SimulationContextGpuImpl(QObject* parent /*= nullptr*/)
	: SimulationContextApi(parent)
{
	_worker = new GpuWorker;
	_worker->moveToThread(&_thread);
	connect(this, &SimulationContextGpuImpl::calculateTimestepWithGpu, _worker, &GpuWorker::calculateTimestep);
	connect(_worker, &GpuWorker::timestepCalculated, this, &SimulationContextGpuImpl::timestepCalculatedWithGpu);
	_thread.start();
}

SimulationContextGpuImpl::~SimulationContextGpuImpl()
{
	_thread.quit();
	_thread.wait();
	delete _worker;
	for (auto const &observer : _observers) {
		observer->unregister();
	}
}

void SimulationContextGpuImpl::init(SpaceMetricApi *metric, SymbolTable *symbolTable, SimulationParameters *parameters)
{
	SET_CHILD(_metric, metric);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_parameters, parameters);
	_worker->init(metric);
}

SpaceMetricApi * SimulationContextGpuImpl::getSpaceMetric() const
{
	return _metric;
}

SymbolTable * SimulationContextGpuImpl::getSymbolTable() const
{
	return _symbolTable;
}

SimulationParameters * SimulationContextGpuImpl::getSimulationParameters() const
{
	return _parameters;
}

void SimulationContextGpuImpl::registerObserver(GpuObserver * observer)
{
	_observers.push_back(observer);
}

void SimulationContextGpuImpl::unregisterObserver(GpuObserver * observer)
{
	_observers.erase(std::remove(_observers.begin(), _observers.end(), observer), _observers.end());
}

void SimulationContextGpuImpl::notifyObserver()
{
	for (auto const &observer : _observers) {
		observer->accessToUnits();
	}
}

GpuWorker * SimulationContextGpuImpl::getGpuWorker()
{
	return _worker;
}

void SimulationContextGpuImpl::calculateTimestep()
{
	Q_EMIT calculateTimestepWithGpu();
}

void SimulationContextGpuImpl::timestepCalculatedWithGpu()
{
	Q_EMIT timestepCalculated();
}

