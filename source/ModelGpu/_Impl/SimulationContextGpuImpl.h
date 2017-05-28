#pragma once

#include "Model/SimulationContextApi.h"

class SimulationContextGpuImpl
	: public SimulationContextApi
{
	Q_OBJECT
public:
	SimulationContextGpuImpl(QObject* parent = nullptr) : SimulationContextApi(parent) {}
	virtual ~SimulationContextGpuImpl() = default;

	void init(SpaceMetricApi *metric, SymbolTable *symbolTable, SimulationParameters *parameters);

	virtual SpaceMetricApi* getSpaceMetric() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;

private:
	SpaceMetricApi *_metric = nullptr;
	SymbolTable *_symbolTable = nullptr;
	SimulationParameters *_parameters = nullptr;
};
