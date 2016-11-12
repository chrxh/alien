#ifndef CELLDECORATORFACTORY_H
#define CELLDECORATORFACTORY_H

#include "constants.h"

#include <QString>

class Grid;
class Cell;
class CellFunction;
class EnergyGuidance;

class CellFeatureFactory
{
public:
    virtual ~CellFeatureFactory () {}

    virtual void addCellFunction (Cell* cell, CellFunctionType type, quint8* data, Grid*& grid) = 0;
    virtual void addCellFunction (Cell* cell, CellFunctionType type, Grid*& grid) = 0;
    virtual void addCellFunction (Cell* cell, CellFunctionType type, QDataStream& stream, Grid*& grid) = 0;

    virtual void addEnergyGuidance (Cell* cell, Grid*& grid) = 0;
};

#endif // CELLDECORATORFACTORY_H
