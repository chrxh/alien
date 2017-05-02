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
	delete _signalMapper;
	for (auto const& ts : _threadsAndCalcSignals) {
		delete ts.thr;
		delete ts.signal;
	}
	_threadsAndCalcSignals.clear();
	_signalMapper = new QSignalMapper(this);
	connect(_signalMapper, static_cast<void(QSignalMapper::*)(QObject*)>(&QSignalMapper::mapped), this, &UnitThreadControllerImpl::threadFinishedCalculation);
}

void UnitThreadControllerImpl::registerUnit(Unit * unit)
{
	auto newThread = new UnitThread(this);
	unit->moveToThread(newThread);
	connect(newThread, &QThread::finished, unit, &QObject::deleteLater);
	_threadsByContexts[unit->getContext()] = newThread;
	
	auto signal = new SignalWrapper(this);
	connect(signal, &SignalWrapper::signal, unit, &Unit::calcNextTimestep);

	connect(unit, &Unit::nextTimestepCalculated, _signalMapper, static_cast<void(QSignalMapper::*)()>(&QSignalMapper::map));
	_signalMapper->setMapping(unit, newThread);

	_threadsAndCalcSignals.push_back({ newThread , signal });
}

void UnitThreadControllerImpl::start()
{
	updateDependencies();
	startThreads();
	searchAndExecuteReadyThreads();
}

void UnitThreadControllerImpl::threadFinishedCalculation(QObject* sender)
{
	if (UnitThread* thr = dynamic_cast<UnitThread*>(sender)) {
		thr->setState(UnitThread::State::Finished);
		searchAndExecuteReadyThreads();
	}
}

void UnitThreadControllerImpl::updateDependencies()
{
	for (auto const& threadByContext : _threadsByContexts) {
		auto context = threadByContext.first;
		auto thr = threadByContext.second;
		auto compartment = context->getMapCompartment();
		for (auto const& neighborContext : compartment->getNeighborContexts()) {
			thr->addDependency(_threadsByContexts[neighborContext]);
		}
	}
}

void UnitThreadControllerImpl::terminateThreads()
{
	for (auto const& ts : _threadsAndCalcSignals) {
		ts.thr->quit();
	}
	for (auto const& ts : _threadsAndCalcSignals) {
		if (!ts.thr->wait(2000)) {
			ts.thr->terminate();
			ts.thr->wait();
		}
	}
}

void UnitThreadControllerImpl::startThreads()
{
	for (auto const& ts : _threadsAndCalcSignals) {
		ts.thr->start();
	}
}

void UnitThreadControllerImpl::searchAndExecuteReadyThreads()
{
	for (auto const& ts : _threadsAndCalcSignals) {
		if (ts.thr->isReady()) {
			ts.signal->emitSignal();
			ts.thr->setState(UnitThread::State::Working);
		}
	}
}
