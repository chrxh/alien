#ifndef SERIALIZATIONFACADE_H
#define SERIALIZATIONFACADE_H

#include "definitions.h"

class SerializationFacade
{
public:
	virtual ~SerializationFacade() {}

    virtual void serializeSimulationContext(SimulationUnitContext* context, QDataStream& stream) const = 0;
    virtual void deserializeSimulationContext(SimulationUnitContext* prevContext, QDataStream& stream) const = 0;

	virtual void serializeSimulationParameters(SimulationParameters* parameters, QDataStream& stream) const = 0;
	virtual SimulationParameters* deserializeSimulationParameters(QDataStream& stream) const = 0;

	virtual void serializeSymbolTable(SymbolTable* symbolTable, QDataStream& stream) const = 0;
	virtual SymbolTable* deserializeSymbolTable(QDataStream& stream) const = 0;

    virtual void serializeCellCluster(CellCluster* cluster, QDataStream& stream) const = 0;
    virtual CellCluster* deserializeCellCluster(QDataStream& stream, SimulationUnitContext* context) const = 0;

    virtual void serializeFeaturedCell(Cell* cell, QDataStream& stream) const = 0;
    virtual Cell* deserializeFeaturedCell(QDataStream& stream, SimulationUnitContext* context) const = 0;

    virtual void serializeEnergyParticle(EnergyParticle* particle, QDataStream& stream) const = 0;
    virtual EnergyParticle* deserializeEnergyParticle(QDataStream& stream, SimulationUnitContext* context) const = 0;

    virtual void serializeToken(Token* token, QDataStream& stream) const = 0;
    virtual Token* deserializeToken(QDataStream& stream, SimulationUnitContext* context) const = 0;

};

#endif //SERIALIZATIONFACADE_H


