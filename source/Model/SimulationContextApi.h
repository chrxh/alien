#ifndef SIMULATIONCONTEXTAPI_H
#define SIMULATIONCONTEXTAPI_H

#include "Model/Definitions.h"

class MODEL_EXPORT SimulationContextApi
	: public QObject
{
Q_OBJECT
public:
	SimulationContextApi(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationContextApi() = default;

	virtual SpaceMetricApi* getSpaceMetric() const = 0;
	virtual SymbolTable* getSymbolTable() const = 0;
	virtual SimulationParameters* getSimulationParameters() const = 0;
};

#endif // SIMULATIONCONTEXTAPI_H
