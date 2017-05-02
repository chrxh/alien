#include <QThread>

#include "model/context/Unit.h"
#include "model/context/UnitContext.h"
#include "model/context/MapCompartment.h"
#include "UnitThreadControllerImpl.h"

UnitThreadControllerImpl::UnitThreadControllerImpl(QObject * parent)
	: UnitThreadController(parent)
{
}

UnitThreadControllerImpl::~UnitThreadControllerImpl()
{
	terminateThreads();
}

void UnitThreadControllerImpl::init(int maxRunningThreads)
{
	terminateThreads();
	_maxRunningThreads = maxRunningThreads;
	for (auto const& thr : _threads) {
		delete thr;
	}
	_threads.clear();

}

void UnitThreadControllerImpl::registerUnit(Unit * unit)
{
	auto newThread = new QThread(this);
	newThread->connect(newThread, &QThread::finished, unit, &QObject::deleteLater);
	unit->moveToThread(newThread);
	_threads.push_back(newThread);
	_threadsByContexts[unit->getContext()] = newThread;
}

void UnitThreadControllerImpl::start()
{
	updateDependencies();
	//newThread->start();
}

void UnitThreadControllerImpl::updateDependencies()
{
	for (auto const& threadByContext : _threadsByContexts) {
		auto context = threadByContext.first;
		auto thr = threadByContext.second;
		auto compartment = context->getMapCompartment();
		auto getThread = [&](MapCompartment::RelativeLocation rel) {
			return _threadsByContexts[compartment->getNeighborContext(rel)];
		};
		_dependencies[thr].push_back(getThread(MapCompartment::RelativeLocation::UpperLeft));
		_dependencies[thr].push_back(getThread(MapCompartment::RelativeLocation::Upper));
		_dependencies[thr].push_back(getThread(MapCompartment::RelativeLocation::UpperRight));
		_dependencies[thr].push_back(getThread(MapCompartment::RelativeLocation::Left));
		_dependencies[thr].push_back(getThread(MapCompartment::RelativeLocation::Right));
		_dependencies[thr].push_back(getThread(MapCompartment::RelativeLocation::LowerLeft));
		_dependencies[thr].push_back(getThread(MapCompartment::RelativeLocation::Lower));
		_dependencies[thr].push_back(getThread(MapCompartment::RelativeLocation::LowerRight));
	}
}

void UnitThreadControllerImpl::terminateThreads()
{
	for (auto const& thr : _threads) {
		thr->quit();
	}
	for (auto const& thr : _threads) {
		if (!thr->wait(2000)) {
			thr->terminate();
			thr->wait();
		}
	}
}
