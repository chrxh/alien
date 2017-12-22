#pragma once

#include <QTime>

#include "Model/Api/SimulationController.h"

class SimulationControllerImpl
	: public SimulationController
{
	Q_OBJECT
public:
	SimulationControllerImpl(QObject* parent = nullptr);
	virtual ~SimulationControllerImpl() = default;

	virtual void init(SimulationContext* context);
	virtual void setRun(bool run) override;
	virtual void calculateSingleTimestep() override;
	virtual SimulationContext* getContext() const override;
	virtual uint getTimestep() const override;
	virtual void setTimestep(uint value) override;

private:
	SimulationContextLocal* _context = nullptr;

	bool _flagRunMode = false;
	QTimer* _oneSecondTimer = nullptr;
	QTime _timeSinceLastStart;
	int _displayedFramesSinceLastStart = 0;
	int _timestep = 0;
};

