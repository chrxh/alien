#pragma once
#include <QObject>

#include "SimulationMonitorCpu.h"
#include "Definitions.h"
#include "UnitObserver.h"

class SimulationMonitorCpuImpl
	: public SimulationMonitorCpu
	, public UnitObserver
{
	Q_OBJECT

public:
	SimulationMonitorCpuImpl(QObject * parent = nullptr);
	virtual ~SimulationMonitorCpuImpl();

	virtual void init(SimulationControllerCpu* controller) override;

	virtual void requireData() override;
	virtual MonitorData const& retrieveData() override;

	//from UnitObserver
	virtual void unregister() override;
	virtual void accessToUnits() override;

private:
	void calcMonitorData();
	void calcMonitorDataForUnit(Unit* unit);

	SimulationContextCpuImpl* _context = nullptr;
	bool _registered = false;

	bool _dataRequired = false;
	MonitorData _data;
};
