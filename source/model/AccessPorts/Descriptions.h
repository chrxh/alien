#ifndef CELLDESCRIPTION_H
#define CELLDESCRIPTION_H

#include "model/Definitions.h"
#include "model/features/CellFeatureEnums.h"

struct CellFunctionDescription
{
	Enums::CellFunction::Type type;
	QByteArray data;
};

struct TokenDescription
{
	qreal energy;
	QByteArray data;
};

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
