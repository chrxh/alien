#ifndef SIMULATIONCONTEXTAPI_H
#define SIMULATIONCONTEXTAPI_H

#include "Model/Definitions.h"

class MODEL_EXPORT SimulationContext
	: public QObject
{
Q_OBJECT
public:
	SimulationContext(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationContext() = default;

	virtual SpaceMetric* getSpaceMetric() const = 0;
	virtual SymbolTable* getSymbolTable() const = 0;
	virtual SimulationParameters* getSimulationParameters() const = 0;
};

#endif // SIMULATIONCONTEXTAPI_H
