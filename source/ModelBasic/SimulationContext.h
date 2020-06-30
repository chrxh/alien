#pragma once

#include "Definitions.h"

class MODELBASIC_EXPORT SimulationContext
	: public QObject
{
Q_OBJECT
public:
	SimulationContext(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationContext() = default;

	virtual SpaceProperties* getSpaceProperties() const = 0;
	virtual SymbolTable* getSymbolTable() const = 0;
	virtual SimulationParameters const& getSimulationParameters() const = 0;
	virtual NumberGenerator* getNumberGenerator() const = 0;	//must be used to generate ids in descriptions
    virtual int getTimestep() const = 0;
    virtual void setTimestep(int timestep) = 0;

	virtual map<string, int> getSpecificData() const = 0;

	virtual void setSimulationParameters(SimulationParameters const& parameters) = 0;
    virtual void setExecutionParameters(ExecutionParameters const& parameters) = 0;
};
