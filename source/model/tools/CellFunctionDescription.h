#ifndef CELLFUNCTIONDESCRIPTION_H
#define CELLFUNCTIONDESCRIPTION_H

#include "model/Definitions.h"
#include "model/features/CellFeatureEnums.h"

struct CellFunctionDescription
{
	Enums::CellFunction::Type cellFunctionType;
	QByteArray cellFunctionData;
};

#endif // CELLFUNCTIONDESCRIPTION_H
