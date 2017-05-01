#include <QThread>

#include "model/context/simulationunit.h"
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
	for (auto const& thread : _threads) {
		delete thread;
	}
	_threads.clear();

}

void SimulationThreadsImpl::registerUnit(SimulationUnit * unit)
{
	auto newThread = new QThread(this);
	newThread->connect(newThread, &QThread::finished, unit, &QObject::deleteLater);
	unit->moveToThread(newThread);
	_threads.push_back(newThread);
}

void SimulationThreadsImpl::start() const
{
	//newThread->start();
}

void SimulationThreadsImpl::terminateThreads()
{
	for (auto const& thread : _threads) {
		thread->quit();
	}
	for (auto const& thread : _threads) {
		if (!thread->wait(2000)) {
			thread->terminate();
			thread->wait();
		}
	}
}
