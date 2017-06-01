#pragma once

#include <QTime>

#include "Model/SimulationController.h"
#include "DefinitionsImpl.h"

class SimulationControllerGpuImpl
	: public SimulationController
{
	Q_OBJECT
public:
	SimulationControllerGpuImpl(QObject* parent = nullptr);
	virtual ~SimulationControllerGpuImpl() = default;

	virtual void init(SimulationContextApi* context) override;
	virtual void setRun(bool run) override;
	virtual void calculateSingleTimestep() override;
	virtual SimulationContextApi* getContext() const override;

private:
	Q_SLOT void oneSecondTimerTimeout();

	SimulationContextGpuImpl *_context = nullptr;

	bool _flagSimulationRunning = false;
	QTimer* _oneSecondTimer = nullptr;
	int _timestepsPerSecond = 0;
	QTime _timeSinceLastStart;
	int _displayedFramesSinceLastStart = 0;
};