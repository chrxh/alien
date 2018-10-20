#pragma once

#include <QThread>

#include "ModelBasic/SimulationContext.h"
#include "DefinitionsImpl.h"

class SimulationContextGpuImpl
	: public SimulationContext
{
	Q_OBJECT
public:
	SimulationContextGpuImpl(QObject* parent = nullptr);
	virtual ~SimulationContextGpuImpl();

	void init(SpaceProperties *metric, SymbolTable *symbolTable, SimulationParameters *parameters);

	virtual SpaceProperties* getSpaceProperties() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;
	virtual NumberGenerator* getNumberGenerator() const override;

	virtual map<string, int> getSpecificData() const override;

	virtual void setSimulationParameters(SimulationParameters* parameters) override;

	virtual ThreadController* getGpuThreadController() const;

private:
	SpaceProperties *_metric = nullptr;
	SymbolTable *_symbolTable = nullptr;
	SimulationParameters *_parameters = nullptr;
	ThreadController *_threadController = nullptr;
	NumberGenerator* _numberGen = nullptr;
};
