#ifndef ENERGY_H
#define ENERGY_H

#include <QVector3D>

class CellCluster;
class Grid;
class EnergyParticle
{
public:
    EnergyParticle (qreal amount_, QVector3D pos_, QVector3D vel_, Grid*& grid);
    EnergyParticle (QDataStream& stream, QMap< quint64, EnergyParticle* >& oldIdEnergyMap, Grid*& grid);

    bool movement (CellCluster*& cluster);

    void serialize (QDataStream& stream);

private:
    Grid*& _grid;

public:
    qreal amount;
    QVector3D pos;
    QVector3D vel;
    quint64 id;
    quint8 color;
};

#endif // ENERGY_H
