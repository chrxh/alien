#ifndef FEATURE_DESCRIPTIONS_H
#define FEATURE_DESCRIPTIONS_H

#include "model/Definitions.h"
#include "model/features/CellFeatureEnums.h"

struct CellFunctionDescription
{
	Enums::CellFunction::Type type = Enums::CellFunction::COMPUTER;
	QByteArray data;

	CellFunctionDescription& setType(Enums::CellFunction::Type value) { type = value; return *this; }
	CellFunctionDescription& setData(QByteArray const &value) { data = value; return *this; }
};

#endif // FEATURE_DESCRIPTIONS_H
