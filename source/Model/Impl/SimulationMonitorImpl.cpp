#include "SimulationMonitorImpl.h"

SimulationMonitorImpl::SimulationMonitorImpl(QObject * parent)
	: SimulationMonitor(parent)
{
	
}

void SimulationMonitorImpl::init(SimulationContext * context)
{
}

void SimulationMonitorImpl::requireData()
{
}

MonitorData const & SimulationMonitorImpl::retrieveData()
{
	return _data;
}
