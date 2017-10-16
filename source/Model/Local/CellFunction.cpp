#include <QString>
#include <QtCore/qmath.h>

#include "Model/Local/Cell.h"
#include "Model/Local/Physics.h"
#include "Model/Local/UnitContext.h"
#include "Model/Local/SpaceMetricLocal.h"

#include "CellFunction.h"


void CellFunction::appendDescriptionImpl(CellFeatureDescription & desc) const
{
	desc.setType(getType());
	desc.setVolatileData(getInternalData());
}

qreal CellFunction::calcAngle (Cell* origin, Cell* ref1, Cell* ref2) const
{
	SpaceMetricLocal* topo = _context->getSpaceMetric();
    QVector2D v1 = topo->displacement(origin->calcPosition(), ref1->calcPosition());
    QVector2D v2 = topo->displacement(origin->calcPosition(), ref2->calcPosition());
    return Physics::clockwiseAngleFromFirstToSecondVector(v1, v2);
}


