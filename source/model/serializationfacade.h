#ifndef SERIALIZATIONFACADE_H
#define SERIALIZATIONFACADE_H

#include "definitions.h"

class SerializationFacade
{
public:
	virtual ~SerializationFacade() {}

    virtual void serializeSimulationContext(SimulationContext* context, QDataStream& stream) const = 0;
    virtual SimulationContext* deserializeSimulationContext(QDataStream& stream) const = 0;

    virtual void serializeCellCluster(CellCluster* cluster, QDataStream& stream) const = 0;
    virtual CellCluster* deserializeCellCluster(QDataStream& stream, SimulationContext* context) const = 0;

    virtual void serializeFeaturedCell(Cell* cell, QDataStream& stream) const = 0;
    virtual Cell* deserializeFeaturedCell(QDataStream& stream, SimulationContext* context) const = 0;

    virtual void serializeEnergyParticle(EnergyParticle* particle, QDataStream& stream) const = 0;
    virtual EnergyParticle* deserializeEnergyParticle(QDataStream& stream, SimulationContext* context) const = 0;

    virtual void serializeToken(Token* token, QDataStream& stream) const = 0;
    virtual Token* deserializeToken(QDataStream& stream) const = 0;

};

#endif //SERIALIZATIONFACADE_H


