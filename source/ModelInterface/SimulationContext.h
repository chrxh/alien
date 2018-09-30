#pragma once

#include "Definitions.h"

class MODELINTERFACE_EXPORT SimulationContext
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
	virtual SimulationParameters* getSimulationParameters() const = 0;
	virtual CellComputerCompiler* getCellComputerCompiler() const = 0;

	virtual void setSimulationParameters(SimulationParameters* parameters) = 0;
};
