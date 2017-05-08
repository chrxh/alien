#include <QString>
#include <QtCore/qmath.h>

#include "model/entities/Cell.h"
#include "model/physics/Physics.h"
#include "model/context/UnitContext.h"
#include "model/context/SpaceMetric.h"

#include "CellFunction.h"


qreal CellFunction::calcAngle (Cell* origin, Cell* ref1, Cell* ref2) const
{
	SpaceMetric* topo = _context->getSpaceMetric();
    QVector2D v1 = topo->displacement(origin->calcPosition(), ref1->calcPosition());
    QVector2D v2 = topo->displacement(origin->calcPosition(), ref2->calcPosition());
    return Physics::clockwiseAngleFromFirstToSecondVector(v1, v2);
}


