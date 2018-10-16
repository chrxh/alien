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
	_mutexForRequirement.lock();
	_requireData = true;
	_mutexForRequirement.unlock();
}

bool CudaBridge::isDataRequired()
{
	bool result;

	_mutexForRequirement.lock();
	result = _requireData;
	_mutexForRequirement.unlock();

	return result;
}

void CudaBridge::dataObtainedIntern()
{
	_mutexForRequirement.lock();
	_requireData = false;
	_mutexForRequirement.unlock();

	Q_EMIT dataObtained();
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
		if (isDataRequired()) {
			if (_mutexForData.try_lock()) {
				_cudaData = cudaGetData();
				_mutexForData.unlock();
				dataObtainedIntern();
			}
		}
	} while (!_stopAfterNextTimestep);
	_simRunning = false;
}
