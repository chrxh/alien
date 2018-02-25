#pragma once
#include <QObject>

#include "Model/Api/SimulationMonitor.h"

class SimulationMonitorImpl
	: public SimulationMonitor
{
	Q_OBJECT

public:
	SimulationMonitorImpl(QObject * parent = nullptr);
	virtual ~SimulationMonitorImpl() = default;

	virtual void init(SimulationContext* context) override;

	virtual void requireData() override;
	virtual MonitorData const& retrieveData() override;

private:
	MonitorData _data;
};
