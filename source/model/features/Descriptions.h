#ifndef FEATURE_DESCRIPTIONS_H
#define FEATURE_DESCRIPTIONS_H

#include "model/Definitions.h"
#include "model/features/CellFeatureEnums.h"

struct CellFunctionDescription
{
	Enums::CellFunction::Type type;
	QByteArray data;

	CellFunctionDescription& setType(Enums::CellFunction::Type t) { type = t; return *this; }
	CellFunctionDescription& setData(QByteArray const &d) { data = d; return *this; }
};

#endif // FEATURE_DESCRIPTIONS_H
