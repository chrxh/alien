#pragma once

#include <QObject>

#include "Definitions.h"
#include "UnitObserver.h"

class SimulationAttributeSetter
	: public QObject
	, public UnitObserver
{
	Q_OBJECT

public:
	SimulationAttributeSetter(QObject * parent = nullptr);
	virtual ~SimulationAttributeSetter();

	virtual void init(SimulationContext * context);

	virtual void setSimulationParameters(SimulationParameters const* parameters);

private:
	void unregister() override;
	void accessToUnits() override;

	UnitThreadController* _threadController = nullptr;
	UnitGrid* _grid = nullptr;

	bool _registered = false;
	bool _updateSimulationParameters = false;

	SimulationParameters const* _parameters;
};
