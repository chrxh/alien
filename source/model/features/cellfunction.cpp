#include "cellfunction.h"

#include "model/entities/grid.h"
#include "model/physics/physics.h"

#include <QString>
#include <QtCore/qmath.h>

qreal CellFunction::calcAngle (Cell* origin, Cell* ref1, Cell* ref2) const
{
    QVector3D v1 = _grid->displacement(origin, ref1);
    QVector3D v2 = _grid->displacement(origin, ref2);
    return Physics::clockwiseAngleFromFirstToSecondVector(v1, v2);
}


