#include <QThread>

#include "model/context/simulationunit.h"
#include "model/context/simulationunitcontext.h"
#include "model/context/mapcompartment.h"
#include "simulationthreadsimpl.h"

SimulationThreadsImpl::SimulationThreadsImpl(QObject * parent)
	: SimulationThreads(parent)
{
}

SimulationThreadsImpl::~SimulationThreadsImpl()
{
	terminateThreads();
}

void SimulationThreadsImpl::init(int maxRunningThreads)
{
	terminateThreads();
	_maxRunningThreads = maxRunningThreads;
	for (auto const& thr : _threads) {
		delete thr;
	}
	_threads.clear();

}

void SimulationThreadsImpl::registerUnit(SimulationUnit * unit)
{
	auto newThread = new QThread(this);
	newThread->connect(newThread, &QThread::finished, unit, &QObject::deleteLater);
	unit->moveToThread(newThread);
	_threads.push_back(newThread);
	_threadsByContexts[unit->getContext()] = newThread;
}

void SimulationThreadsImpl::start()
{
	updateDependencies();
	//newThread->start();
}

void SimulationThreadsImpl::updateDependencies()
{
	for (auto const& threadByContext : _threadsByContexts) {
		auto context = threadByContext.first;
		auto thr = threadByContext.second;
		auto compartment = context->getMapCompartment();
		auto getThread = [&](MapCompartment::RelativeLocation location) {
			return _threadsByContexts[compartment->getNeighborContext(location)];
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

void SimulationThreadsImpl::terminateThreads()
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
