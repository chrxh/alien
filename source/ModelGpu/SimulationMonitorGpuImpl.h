#pragma once

#include "SimulationMonitorGpu.h"

class SimulationMonitorGpuImpl
	: public SimulationMonitorGpu
{
	Q_OBJECT
public:
	SimulationMonitorGpuImpl(QObject* parent = nullptr) : SimulationMonitorGpu(parent) {}
	virtual ~SimulationMonitorGpuImpl() = default;

	virtual void init(SimulationControllerGpu* controller) override;

	virtual void requireData() override;
	virtual MonitorData const& retrieveData() override;

private:
	MonitorData _data;
};

