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

	virtual void init(SimulationContext* context, uint timestep);
	virtual void setRun(bool run) override;
	virtual void calculateSingleTimestep() override;
	virtual SimulationContext* getContext() const override;
	virtual uint getTimestep() const override;
	virtual void setRestrictTimestepsPreSecond(optional<int> tps) override;

private:
	Q_SLOT void nextTimestepCalculatedIntern();
	Q_SLOT void restrictTpsTimerTimeout();

	SimulationContextLocal* _context = nullptr;

	bool _runMode = false;
	bool _calculationRunning = false;
	bool _triggerNewTimestep = false;
	QTimer* _restrictTpsTimer = nullptr;
	QTime _timeSinceLastStart;
	int _displayedFramesSinceLastStart = 0;
	uint _timestep = 0;
	optional<int> _restrictTps;
};

