#include <functional>
#include <QImage>
#include <QElapsedTimer>
#include <QThread>

#include "ModelBasic/SpaceProperties.h"
#include "CudaInterface.cuh"

#include "CudaWorker.h"

CudaWorker::~CudaWorker()
{
	delete _simDataManager;
}

void CudaWorker::init(SpaceProperties* spaceProp)
{
	_spaceProp = spaceProp;
	auto size = spaceProp->getSize();
	delete _simDataManager;
	_simDataManager = new CudaSimulator({ size.x, size.y });
}

void CudaWorker::requireData()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_requireData = true;

	if (!_simRunning) {
		_mutexForData.lock();
		_cudaData = _simDataManager->getDataForAccess();
		_mutexForData.unlock();
		_requireData = false;
		Q_EMIT dataObtained();
	}
}

bool CudaWorker::isDataRequired()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	return _requireData;
}

void CudaWorker::requireDataFinished()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_requireData = false;
}

bool CudaWorker::isDataUpdated()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	return _updateData;
}

void CudaWorker::updateDataFinished()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_updateData = false;
}

optional<int> CudaWorker::getTps()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	return _tps;
}

bool CudaWorker::stopAfterNextTimestep()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	return _stopAfterNextTimestep;
}

void CudaWorker::setSimulationRunning(bool running)
{
	_mutexForFlags.lock();
	_simRunning = running;
	_mutexForFlags.unlock();
}

bool CudaWorker::isSimulationRunning()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	return _simRunning;
}

void CudaWorker::lockData()
{
	_mutexForData.lock();
}

void CudaWorker::unlockData()
{
	_mutexForData.unlock();
}

SimulationDataForAccess& CudaWorker::retrieveData()
{
	return _cudaData;
}

void CudaWorker::updateData()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_updateData = true;

	if (!_simRunning) {
		_simDataManager->setDataForAccess(_cudaData);
		_updateData = false;
	}
}

void CudaWorker::stopAfterNextTimestep(bool value)
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_stopAfterNextTimestep = value;
}

void CudaWorker::restrictTimestepsPerSecond(optional<int> tps)
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_tps = tps;
}


void CudaWorker::runSimulation()
{
	QElapsedTimer timer;
	setSimulationRunning(true);
	do {
		timer.start();
		if (isDataUpdated()) {
			if (_mutexForData.try_lock()) {
				_simDataManager->setDataForAccess(_cudaData);
				updateDataFinished();
				_mutexForData.unlock();
			}
		}

		_simDataManager->calcNextTimestep();

		Q_EMIT timestepCalculated();

		if (isDataRequired()) {
			if (_mutexForData.try_lock()) {
				_cudaData = _simDataManager->getDataForAccess();
				requireDataFinished();
				_mutexForData.unlock();
				Q_EMIT dataObtained();
			}
		}
		if (auto const& tps = getTps()) {
			int remainingTime = 1000000/(*tps) - timer.nsecsElapsed()/1000;
			if (remainingTime > 0) {
				QThread::usleep(remainingTime);
			}
		}
	} while (!stopAfterNextTimestep());
	setSimulationRunning(false);
}
