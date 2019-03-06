#pragma once

#include "SimulationMonitorGpu.h"
#include "DefinitionsImpl.h"

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
	Q_SLOT void dataObtainedFromGpu();

	void calcMonitorData(SimulationAccessTO* access);

private:
	list<QMetaObject::Connection> _connections;

	SimulationContextGpuImpl* _context = nullptr;
	MonitorData _data;
};

