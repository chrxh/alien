#pragma once

#include "Definitions.h"
#include "Descriptions.h"
#include "MonitorData.h"

class ENGINEINTERFACE_EXPORT SimulationMonitor
	: public QObject
{
	Q_OBJECT
public:
	SimulationMonitor(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationMonitor() = default;

	virtual void requireData() = 0;
	Q_SIGNAL void dataReadyToRetrieve();
	virtual MonitorData const& retrieveData() = 0;
};

