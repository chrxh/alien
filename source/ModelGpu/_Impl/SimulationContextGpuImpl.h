#pragma once

#include <QThread>

#include "Model/Api/SimulationContext.h"
#include "DefinitionsImpl.h"

class SimulationContextGpuImpl
	: public SimulationContext
{
	Q_OBJECT
public:
	SimulationContextGpuImpl(QObject* parent = nullptr);
	virtual ~SimulationContextGpuImpl();

	void init(SpaceProperties *metric, SymbolTable *symbolTable, SimulationParameters *parameters);

	virtual SpaceProperties* getSpaceMetric() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;
	virtual ThreadController* getGpuThreadController() const;

private:
	SpaceProperties *_metric = nullptr;
	SymbolTable *_symbolTable = nullptr;
	SimulationParameters *_parameters = nullptr;
	ThreadController *_threadController = nullptr;
};
