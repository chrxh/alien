#ifndef ENERGY_H
#define ENERGY_H

#include <QVector3D>

#include "model/definitions.h"

class EnergyParticle
{
public:
    EnergyParticle (SimulationContext* context);
    EnergyParticle (qreal amount_, QVector3D pos_, QVector3D vel_, SimulationContext* context);

    bool movement (CellCluster*& cluster);

    void serializePrimitives (QDataStream& stream);
    void deserializePrimitives (QDataStream& stream);

public:
    qreal amount = 0.0;
    QVector3D pos;
    QVector3D vel;
    quint64 id = 0;
    quint8 color = 0;

private:
    SimulationContext* _context;
	Topology* _topology;
	CellMap* _cellMap;
	EnergyParticleMap* _energyMap;
};

#endif // ENERGY_H
