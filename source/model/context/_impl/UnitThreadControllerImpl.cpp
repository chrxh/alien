#include "model/context/Unit.h"
#include "model/context/UnitContext.h"
#include "model/context/MapCompartment.h"

#include "UnitThreadControllerImpl.h"
#include "UnitThread.h"

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
	auto newThread = new UnitThread(this);
	connect(newThread, &QThread::finished, unit, &QObject::deleteLater);
	unit->moveToThread(newThread);
	_threads.push_back(newThread);
	_threadsByContexts[unit->getContext()] = newThread;
}

void UnitThreadControllerImpl::start()
{
	updateDependencies();
	for (auto const& thr : _threads) {
		thr->start();
	}
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
		thr->addDependency(getThread(MapCompartment::RelativeLocation::UpperLeft));
		thr->addDependency(getThread(MapCompartment::RelativeLocation::Upper));
		thr->addDependency(getThread(MapCompartment::RelativeLocation::UpperRight));
		thr->addDependency(getThread(MapCompartment::RelativeLocation::Left));
		thr->addDependency(getThread(MapCompartment::RelativeLocation::Right));
		thr->addDependency(getThread(MapCompartment::RelativeLocation::LowerLeft));
		thr->addDependency(getThread(MapCompartment::RelativeLocation::Lower));
		thr->addDependency(getThread(MapCompartment::RelativeLocation::LowerRight));
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
