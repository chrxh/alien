#pragma once

#include <QTime>

#include "SimulationControllerGpu.h"
#include "DefinitionsImpl.h"

class SimulationControllerGpuImpl
	: public SimulationControllerGpu
{
	Q_OBJECT
public:
	SimulationControllerGpuImpl(QObject* parent = nullptr);
	virtual ~SimulationControllerGpuImpl() = default;

	virtual void init(SimulationContext* context);
	virtual void setRun(bool run) override;
	virtual void calculateSingleTimestep() override;
	virtual SimulationContext* getContext() const override;
	virtual void setRestrictTimestepsPerSecond(optional<int> tps) override;
    virtual void setEnableCalculateFrames(bool enabled) override;

private:
	Q_SLOT void oneSecondTimerTimeout();
	Q_SLOT void frameTimerTimeout();

	SimulationContextGpuImpl *_context = nullptr;

	RunningMode _mode = RunningMode::DoNothing;
	QTime _timeSinceLastStart;
	int _timestepsPerSecond = 0;
	int _displayedFramesSinceLastStart = 0;
	QTimer* _frameTimer = nullptr;
	QTimer* _oneSecondTimer = nullptr;
};