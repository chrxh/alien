#include <functional>
#include <QImage>

#include "ModelBasic/SpaceProperties.h"
#include "CudaInterface.cuh"

#include "CudaBridge.h"

CudaBridge::~CudaBridge()
{
	cudaShutdown();
}

void CudaBridge::init(SpaceProperties* spaceProp)
{
	_spaceProp = spaceProp;
	auto size = spaceProp->getSize();
	cudaInit({ size.x, size.y });
}

void CudaBridge::requireData()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_requireData = true;

	if (!_simRunning) {
		_mutexForData.lock();
		_cudaData = cudaGetData();
		_mutexForData.unlock();
		_requireData = false;
		Q_EMIT dataObtained();
	}
}

bool CudaBridge::isDataRequired()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	return _requireData;
}

void CudaBridge::requireDataFinished()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_requireData = false;
}

bool CudaBridge::isDataUpdated()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	return _updateData;
}

void CudaBridge::updateDataFinished()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_updateData = false;
}

bool CudaBridge::stopAfterNextTimestep()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	return _stopAfterNextTimestep;
}

void CudaBridge::setSimulationRunning(bool running)
{
	_mutexForFlags.lock();
	_simRunning = running;
	_mutexForFlags.unlock();
}

bool CudaBridge::isSimulationRunning()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	return _simRunning;
}

void CudaBridge::lockData()
{
	_mutexForData.lock();
}

void CudaBridge::unlockData()
{
	_mutexForData.unlock();
}

SimulationDataForAccess& CudaBridge::retrieveData()
{
	return _cudaData;
}

void CudaBridge::updateData()
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_updateData = true;

	if (!_simRunning) {
		cudaSetData(_cudaData);
		_updateData = false;
	}
}

void CudaBridge::setFlagStopAfterNextTimestep(bool value)
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_stopAfterNextTimestep = value;
}


void CudaBridge::runSimulation()
{
	setSimulationRunning(true);
	do {
		if (isDataUpdated()) {
			if (_mutexForData.try_lock()) {
				cudaSetData(_cudaData);
				updateDataFinished();
				_mutexForData.unlock();
			}
		}

		cudaCalcNextTimestep();

		Q_EMIT timestepCalculated();

		if (isDataRequired()) {
			if (_mutexForData.try_lock()) {
				_cudaData = cudaGetData();
				requireDataFinished();
				_mutexForData.unlock();
				Q_EMIT dataObtained();
			}
		}
	} while (!stopAfterNextTimestep());
	setSimulationRunning(false);
}
