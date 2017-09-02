#pragma once

#include "Model/Definitions.h"
#include "Model/Features/CellFeatureEnums.h"

struct CellFunctionDescription
{
	Enums::CellFunction::Type type = Enums::CellFunction::COMPUTER;
	QByteArray data;

	CellFunctionDescription& setType(Enums::CellFunction::Type value) { type = value; return *this; }
	CellFunctionDescription& setData(QByteArray const &value) { data = value; return *this; }
	bool operator==(CellFunctionDescription const& other) const {
		return type == other.type
			&& data == other.data;
	}
	bool operator!=(CellFunctionDescription const& other) const { return !operator==(other); }
};

