#ifndef CELLDESCRIPTION_H
#define CELLDESCRIPTION_H

#include "model/Definitions.h"
#include "TokenDescription.h"
#include "CellFunctionDescription.h"

struct CellDescription
{
	QVector3D pos;
	QVector3D vel;
	qreal energy = 0.0;
	int numConnections = 0;
	int maxConnections = 0;
	bool allowToken = true;
	int tokenAccessNumber = 0;
	CellMetadata metadata;

	CellFunctionDescription cellFunction;

	std::vector<TokenDescription> tokens;
};

#endif // CELLDESCRIPTION_H
