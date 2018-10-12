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
	virtual IntVector2D getGridSize() const override;
	virtual uint getMaxThreads() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;
	virtual CellComputerCompiler* getCellComputerCompiler() const override;

	virtual void setSimulationParameters(SimulationParameters* parameters) override;

	virtual ThreadController* getGpuThreadController() const;

private:
	SpaceProperties *_metric = nullptr;
	SymbolTable *_symbolTable = nullptr;
	SimulationParameters *_parameters = nullptr;
	ThreadController *_threadController = nullptr;
};
