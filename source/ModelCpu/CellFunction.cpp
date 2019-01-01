#include <QString>
#include <QtCore/qmath.h>

#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/Physics.h"
#include "Cell.h"
#include "UnitContext.h"

#include "CellFunction.h"

QByteArray CellFunction::getInternalData() const
{
	int memorySize = _context->getSimulationParameters()->cellFunctionComputerCellMemorySize;
	return QByteArray(memorySize, 0);
}

void CellFunction::appendDescriptionImpl(CellFeatureDescription & desc) const
{
	desc.setType(getType());
	desc.setVolatileData(getInternalData());
}

qreal CellFunction::calcAngle (Cell* origin, Cell* ref1, Cell* ref2) const
{
	SpaceProperties* spaceProp = new SpaceProperties();
    QVector2D v1 = spaceProp->displacement(origin->calcPosition(), ref1->calcPosition());
    QVector2D v2 = spaceProp->displacement(origin->calcPosition(), ref2->calcPosition());
    return Physics::clockwiseAngleFromFirstToSecondVector(v1, v2);
}


