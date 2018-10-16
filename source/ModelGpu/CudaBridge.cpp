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

void CudaBridge::setRequireData(bool require)
{
	std::lock_guard<std::mutex> lock(_mutexForFlags);
	_requireData = require;

	if (require && !_simRunning) {
		_mutexForData.lock();
		_cudaData = cudaGetData();
		_mutexForData.unlock();
		_requireData = false;
		Q_EMIT dataObtained();
	}
}

bool CudaBridge::isDataRequired()
{
	bool result;
	_mutexForFlags.lock();
	result = _requireData;
	_mutexForFlags.unlock();
	return result;
}

void CudaBridge::setSimulationRunning(bool running)
{
	_mutexForFlags.lock();
	_simRunning = running;
	_mutexForFlags.unlock();
}

bool CudaBridge::isSimulationRunning()
{
	bool result;
	_mutexForFlags.lock();
	result = _simRunning;
	_mutexForFlags.unlock();
	return result;
}


SimulationDataForAccess CudaBridge::retrieveData()
{
	return _cudaData;
}

void CudaBridge::lockData()
{
	_mutexForData.lock();
}

void CudaBridge::unlockData()
{
	_mutexForData.unlock();
}

void CudaBridge::setFlagStopAfterNextTimestep(bool value)
{
	_stopAfterNextTimestep = value;
}


void CudaBridge::runSimulation()
{
	setSimulationRunning(true);
	do {
		cudaCalcNextTimestep();
		Q_EMIT timestepCalculated();
		if (isDataRequired()) {
			if (_mutexForData.try_lock()) {
				_cudaData = cudaGetData();
				_mutexForData.unlock();
				setRequireData(false);
				Q_EMIT dataObtained();
			}
		}
	} while (!_stopAfterNextTimestep);
	setSimulationRunning(false);
}
