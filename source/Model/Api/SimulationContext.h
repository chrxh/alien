#pragma once

#include "Model/Api/Definitions.h"

class MODEL_EXPORT SimulationContext
	: public QObject
{
Q_OBJECT
public:
	SimulationContext(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationContext() = default;

	virtual SpaceProperties* getSpaceProperties() const = 0;
	virtual IntVector2D getGridSize() const = 0;
	virtual uint getMaxThreads() const = 0;
	virtual SymbolTable* getSymbolTable() const = 0;
	virtual SimulationParameters const* getSimulationParameters() const = 0;
	virtual CellComputerCompiler* getCellComputerCompiler() const = 0;

	virtual void setSimulationParameters(SimulationParameters const* parameters) = 0;
};
