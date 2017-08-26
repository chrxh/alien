#ifndef SERIALIZATIONFACADE_H
#define SERIALIZATIONFACADE_H

#include "Definitions.h"

class SerializationFacade
{
public:
	virtual ~SerializationFacade() {}

    virtual void serializeSimulationContext(UnitContext* context, QDataStream& stream) const = 0;
    virtual void deserializeSimulationContext(UnitContext* prevContext, QDataStream& stream) const = 0;

	virtual void serializeSimulationParameters(SimulationParameters* parameters, QDataStream& stream) const = 0;
	virtual SimulationParameters* deserializeSimulationParameters(QDataStream& stream) const = 0;

	virtual void serializeSymbolTable(SymbolTable* symbolTable, QDataStream& stream) const = 0;
	virtual SymbolTable* deserializeSymbolTable(QDataStream& stream) const = 0;

    virtual void serializeCellCluster(Cluster* cluster, QDataStream& stream) const = 0;
    virtual Cluster* deserializeCellCluster(QDataStream& stream, UnitContext* context) const = 0;

    virtual void serializeFeaturedCell(Cell* cell, QDataStream& stream) const = 0;
    virtual Cell* deserializeFeaturedCell(QDataStream& stream, UnitContext* context) const = 0;

    virtual void serializeEnergyParticle(Particle* particle, QDataStream& stream) const = 0;
    virtual Particle* deserializeEnergyParticle(QDataStream& stream, UnitContext* context) const = 0;

    virtual void serializeToken(Token* token, QDataStream& stream) const = 0;
    virtual Token* deserializeToken(QDataStream& stream, UnitContext* context) const = 0;

};

#endif //SERIALIZATIONFACADE_H


