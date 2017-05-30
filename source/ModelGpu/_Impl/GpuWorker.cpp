#include <functional>

#include "GpuWorker.h"

extern void init_Cuda();
extern void calcNextTimestep_Cuda();
extern void end_Cuda();

GpuWorker::~GpuWorker()
{
	end_Cuda();
}

void GpuWorker::init()
{
	init_Cuda();
}

void GpuWorker::calculateTimestep()
{
	calcNextTimestep_Cuda();
	Q_EMIT timestepCalculated();
}
