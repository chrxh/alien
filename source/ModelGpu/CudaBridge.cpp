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

void CudaBridge::requireAccess()
{
	lockData();
	_requireData = true;
	unlockData();
}

SimulationDataForAccess CudaBridge::retrieveData()
{
	return _cudaData;
}

void CudaBridge::lockData()
{
	_mutex.lock();
}

void CudaBridge::unlockData()
{
	_mutex.unlock();
}

bool CudaBridge::isSimulationRunning()
{
	return _simRunning;
}

void CudaBridge::setFlagStopAfterNextTimestep(bool value)
{
	_stopAfterNextTimestep = value;
}


void CudaBridge::runSimulation()
{
	_simRunning = true;
	do {
		cudaCalcNextTimestep();
		Q_EMIT timestepCalculated();
		if (_mutex.try_lock()) {
			if (_requireData) {
				_cudaData = cudaGetData();
				_requireData = false;
			}
			_mutex.unlock();
			Q_EMIT dataAccessGranted();
		}
	} while (!_stopAfterNextTimestep);
	_simRunning = false;
}
