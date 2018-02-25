#pragma once
#include <QObject>

#include "Gui/Definitions.h"

class MonitorController
	: public QObject
{
	Q_OBJECT

public:
	MonitorController(QObject * parent = nullptr);
	virtual ~MonitorController() = default;

	virtual void init(SimulationContext* context);

private:
	
};
