#pragma once

#include <QThread>

#include "Model/SimulationController.h"
#include "DefinitionsImpl.h"

class SimulationControllerGpuImpl
	: public SimulationController
{
	Q_OBJECT
public:
	SimulationControllerGpuImpl(QObject* parent = nullptr);
	virtual ~SimulationControllerGpuImpl();

	virtual void init(SimulationContextApi* context) override;
	virtual void setRun(bool run) override;
	virtual void calculateSingleTimestep() override;
	virtual SimulationContextApi* getContext() const override;

private:
	Q_SIGNAL void calculateTimestepWithGpu();
	Q_SLOT void nextTimestepCalculatedWithGpu();

	SimulationContextGpuImpl *_context = nullptr;

	QThread _thread;
	GpuWorker* _worker = nullptr;

	bool _flagSimulationRunning = false;
};