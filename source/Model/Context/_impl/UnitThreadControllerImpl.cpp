#include "Model/Context/Unit.h"
#include "Model/Context/UnitContext.h"
#include "Model/Context/MapCompartment.h"
#include "Model/Context/UnitObserver.h"

#include "UnitThreadControllerImpl.h"
#include "UnitThread.h"

UnitThreadControllerImpl::UnitThreadControllerImpl(QObject * parent)
	: UnitThreadController(parent)
{
}

UnitThreadControllerImpl::~UnitThreadControllerImpl()
{
	terminateThreads();
	for (auto const& ts : _threadsAndCalcSignals) {
		delete ts.unit;
	}
	for (auto const &observer : _observers) {
		observer->unregister();
	}
}

void UnitThreadControllerImpl::init(int maxRunningThreads)
{
	terminateThreads();
	_maxRunningThreads = maxRunningThreads;
	delete _signalMapper;
	for (auto const& ts : _threadsAndCalcSignals) {
		delete ts.thr;
		delete ts.calcSignal;
	}
	_threadsAndCalcSignals.clear();
	_signalMapper = new QSignalMapper(this);
	connect(_signalMapper, static_cast<void(QSignalMapper::*)(QObject*)>(&QSignalMapper::mapped), this, &UnitThreadControllerImpl::threadFinishedCalculation);
}

void UnitThreadControllerImpl::registerUnit(Unit * unit)
{
	auto newThread = new UnitThread(this);
	unit->moveToThread(newThread);
	_threadsByContexts[unit->getContext()] = newThread;
	
	auto signal = new SignalWrapper(this);
	connect(signal, &SignalWrapper::signal, unit, &Unit::calculateTimestep);

	connect(unit, &Unit::timestepCalculated, _signalMapper, static_cast<void(QSignalMapper::*)()>(&QSignalMapper::map));
	_signalMapper->setMapping(unit, newThread);

	_threadsAndCalcSignals.push_back({ unit, newThread, signal });
}

void UnitThreadControllerImpl::start()
{
	updateDependencies();
	setAllUnitsReady();
	startThreads();
}

void UnitThreadControllerImpl::registerObserver(UnitObserver * observer)
{
	_observers.push_back(observer);
}

void UnitThreadControllerImpl::unregisterObserver(UnitObserver * observer)
{
	_observers.erase(std::remove(_observers.begin(), _observers.end(), observer), _observers.end());
}

bool UnitThreadControllerImpl::calculateTimestep()
{
	if (isNoThreadWorking()) {
		setAllUnitsReady();
		searchAndExecuteReadyThreads();
		return true;
	}
	return false;
}


void UnitThreadControllerImpl::threadFinishedCalculation(QObject* sender)
{
	if (UnitThread* thr = dynamic_cast<UnitThread*>(sender)) {
		thr->setState(UnitThread::State::Finished);
		--_runningThreads;
		if (areAllThreadsFinished()) {
			notifyObservers();
			Q_EMIT timestepCalculated();
		}
		else {
			searchAndExecuteReadyThreads();
		}
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

bool UnitThreadControllerImpl::areAllThreadsFinished() const
{
	bool result = true;
	for (auto const& ts : _threadsAndCalcSignals) {
		if (!ts.thr->isFinished()) {
			result = false;
			break;
		}
	}
	return result;
}

bool UnitThreadControllerImpl::isNoThreadWorking() const
{
	bool result = true;
	for (auto const& ts : _threadsAndCalcSignals) {
		if (ts.thr->isWorking()) {
			result = false;
			break;
		}
	}
	return result;
}

void UnitThreadControllerImpl::setAllUnitsReady()
{
	for (auto const& ts : _threadsAndCalcSignals) {
		ts.thr->setState(UnitThread::State::Ready);
	}
}

void UnitThreadControllerImpl::searchAndExecuteReadyThreads()
{
	for (auto const& ts : _threadsAndCalcSignals) {
		if (ts.thr->isReady()) {
			ts.calcSignal->emitSignal();
			ts.thr->setState(UnitThread::State::Working);
			if (++_runningThreads == _maxRunningThreads) {
				return;
			}
		}
	}
}

void UnitThreadControllerImpl::notifyObservers() const
{
	for (auto const &observer : _observers) {
		observer->accessToUnits();
	}
}
