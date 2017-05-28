#include <QString>
#include <QtCore/qmath.h>

#include "Model/Entities/Cell.h"
#include "Model/Physics/Physics.h"
#include "Model/Context/UnitContext.h"
#include "Model/Context/SpaceMetric.h"

#include "CellFunction.h"


qreal CellFunction::calcAngle (Cell* origin, Cell* ref1, Cell* ref2) const
{
	SpaceMetric* topo = _context->getSpaceMetric();
    QVector2D v1 = topo->displacement(origin->calcPosition(), ref1->calcPosition());
    QVector2D v2 = topo->displacement(origin->calcPosition(), ref2->calcPosition());
    return Physics::clockwiseAngleFromFirstToSecondVector(v1, v2);
}


