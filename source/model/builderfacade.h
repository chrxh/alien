#ifndef FACTORYFACADE_H
#define FACTORYFACADE_H

#include <QtGlobal>
#include <QVector2D>

#include "model/entities/CellTO.h"
#include "model/features/CellFeatureEnums.h"
#include "model/entities/Descriptions.h"
#include "model/entities/LightDescriptions.h"

#include "Definitions.h"

class BuilderFacade
{
public:
	virtual ~BuilderFacade() = default;

	virtual SimulationContextApi* buildSimulationContext(int maxRunngingThreads, IntVector2D gridSize, SpaceMetric* metric
		, SymbolTable* symbolTable, SimulationParameters* parameters) const = 0;
	virtual SimulationFullAccess* buildSimulationFullAccess(SimulationContextApi* context) const = 0;
	virtual SimulationLightAccess* buildSimulationLightAccess(SimulationContextApi* context) const = 0;
	virtual SimulationController* buildSimulationController(SimulationContextApi* context) const = 0;
	virtual SpaceMetric* buildSpaceMetric(IntVector2D universeSize) const = 0;
	virtual SymbolTable* buildDefaultSymbolTable() const = 0;
	virtual SimulationParameters* buildDefaultSimulationParameters() const = 0;
};

#endif // FACTORYFACADE_H
