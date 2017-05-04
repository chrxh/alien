#ifndef CELLDESCRIPTION_H
#define CELLDESCRIPTION_H

#include "model/Definitions.h"
#include "TokenDescription.h"
#include "CellFunctionDescription.h"

struct CellDescription
{
	qreal clusterAngle = 0.0;
	qreal clusterAngVel = 0.0;
	QVector3D cellPos;
	qreal cellEnergy = 0.0;
	int cellNumCon = 0;
	int cellMaxCon = 0;
	bool cellAllowToken = true;
	int cellTokenAccessNum = 0;
	CellMetadata metadata;

	CellFunctionDescription cellFunction;

	std::vector<TokenDescription> tokens;
};

#endif // CELLDESCRIPTION_H
