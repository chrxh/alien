#ifndef FEATURE_DESCRIPTIONS_H
#define FEATURE_DESCRIPTIONS_H

#include "model/Definitions.h"
#include "model/features/CellFeatureEnums.h"

struct CellFunctionDescription
{
	Tracker<Enums::CellFunction::Type> type;
	Tracker<QByteArray> data;

	CellFunctionDescription& setType(Enums::CellFunction::Type value) { type.init(value); return *this; }
	CellFunctionDescription& setData(QByteArray const &value) { data.init(value); return *this; }
};

#endif // FEATURE_DESCRIPTIONS_H
