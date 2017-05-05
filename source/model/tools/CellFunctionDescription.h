#ifndef CELLFUNCTIONDESCRIPTION_H
#define CELLFUNCTIONDESCRIPTION_H

#include "model/Definitions.h"
#include "model/features/CellFeatureEnums.h"

struct CellFunctionDescription
{
	Enums::CellFunction::Type type;
	QByteArray data;
};

#endif // CELLFUNCTIONDESCRIPTION_H
