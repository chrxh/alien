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
	virtual GpuThreadController* getGpuThreadController() const;

private:
	SpaceMetricApi *_metric = nullptr;
	SymbolTable *_symbolTable = nullptr;
	SimulationParameters *_parameters = nullptr;
	GpuThreadController *_threadController = nullptr;
};
