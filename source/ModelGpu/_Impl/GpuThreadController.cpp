#include "GpuWorker.h"
#include "GpuObserver.h"

#include "GpuThreadController.h"

GpuThreadController::GpuThreadController(QObject* parent /*= nullptr*/)
	: QObject(parent)
{
	_worker = new GpuWorker;
	_worker->moveToThread(&_thread);
	connect(this, &GpuThreadController::calculateTimestepWithGpu, _worker, &GpuWorker::calculateTimestep);
	connect(_worker, &GpuWorker::timestepCalculated, this, &GpuThreadController::timestepCalculatedWithGpu);
	_thread.start();
}

GpuThreadController::~GpuThreadController()
{
	_thread.quit();
	_thread.wait();
	delete _worker;
	for (auto const &observer : _observers) {
		observer->unregister();
	}
}

void GpuThreadController::init(SpaceMetricApi *metric)
{
	_worker->init(metric);
}

void GpuThreadController::registerObserver(GpuObserver * observer)
{
	_observers.push_back(observer);
}

void GpuThreadController::unregisterObserver(GpuObserver * observer)
{
	_observers.erase(std::remove(_observers.begin(), _observers.end(), observer), _observers.end());
}

void GpuThreadController::notifyObserver()
{
	for (auto const &observer : _observers) {
		observer->accessToUnits();
	}
}

GpuWorker * GpuThreadController::getGpuWorker() const
{
	return _worker;
}

bool GpuThreadController::isGpuThreadWorking() const
{
	return _gpuThreadWorking;
}

void GpuThreadController::calculateTimestep()
{
	_gpuThreadWorking = true;
	Q_EMIT calculateTimestepWithGpu();
}

void GpuThreadController::timestepCalculatedWithGpu()
{
	_gpuThreadWorking = false;
	notifyObserver();
	Q_EMIT timestepCalculated();
}

