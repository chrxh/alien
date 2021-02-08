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

	void init(SimulationContext* context);
    bool getRun() override;
    void setRun(bool run) override;
	void calculateSingleTimestep() override;
	SimulationContext* getContext() const override;
	void setRestrictTimestepsPerSecond(optional<int> tps) override;
    void setEnableCalculateFrames(bool enabled) override;

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