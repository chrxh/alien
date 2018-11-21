#include <functional>
#include <QImage>

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
	_simDataManager = new SimulationDataManager({ size.x, size.y });
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

void CudaWorker::setFlagStopAfterNextTimestep(bool value)
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_stopAfterNextTimestep = value;
}


void CudaWorker::runSimulation()
{
	setSimulationRunning(true);
	do {
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
	} while (!stopAfterNextTimestep());
	setSimulationRunning(false);
}
