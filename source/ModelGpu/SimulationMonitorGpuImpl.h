#pragma once

#include "SimulationMonitorGpu.h"
#include "DefinitionsImpl.h"
#include "CudaInterface.cuh"

class SimulationMonitorGpuImpl
	: public SimulationMonitorGpu
{
	Q_OBJECT
public:
	SimulationMonitorGpuImpl(QObject* parent = nullptr);
	virtual ~SimulationMonitorGpuImpl();

	virtual void init(SimulationControllerGpu* controller) override;

	virtual void requireData() override;
	virtual MonitorData const& retrieveData() override;

private:
	Q_SLOT void jobsFinished();

	void calcMonitorData(DataAccessTO const& dataTO);
	string getObjectId() const;

private:
	list<QMetaObject::Connection> _connections;

	SimulationContextGpuImpl* _context = nullptr;
	MonitorData _monitorData;
	DataAccessTO _dataTO;
};

