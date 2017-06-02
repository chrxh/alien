#pragma once

#include <QThread>

#include "Model/SimulationContextApi.h"
#include "DefinitionsImpl.h"

class SimulationContextGpuImpl
	: public SimulationContextApi
{
	Q_OBJECT
public:
	SimulationContextGpuImpl(QObject* parent = nullptr);
	virtual ~SimulationContextGpuImpl();

	void init(SpaceMetricApi *metric, SymbolTable *symbolTable, SimulationParameters *parameters);

	virtual SpaceMetricApi* getSpaceMetric() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;

	virtual void registerObserver(GpuObserver* observer);
	virtual void unregisterObserver(GpuObserver* observer);
	virtual void notifyObserver();
	virtual GpuWorker* getGpuWorker() const;
	virtual bool isGpuThreadWorking() const;

	void calculateTimestep();
	Q_SIGNAL void timestepCalculated();

private:
	Q_SIGNAL void calculateTimestepWithGpu();
	Q_SLOT void timestepCalculatedWithGpu();

	SpaceMetricApi *_metric = nullptr;
	SymbolTable *_symbolTable = nullptr;
	SimulationParameters *_parameters = nullptr;

	QThread _thread;
	GpuWorker* _worker = nullptr;
	bool _gpuThreadWorking = false;

	vector<GpuObserver*> _observers;
};
