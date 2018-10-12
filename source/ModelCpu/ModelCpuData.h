#pragma once

#include "Definitions.h"

class MODELCPU_EXPORT ModelCpuData
{
public:
	ModelCpuData(map<string, int> const& data);
	ModelCpuData(uint maxRunningThreads, IntVector2D const& gridSize);

	uint getMaxRunningThreads() const;
	IntVector2D getGridSize() const;

	map<string, int> getData() const;

private:
	map<string, int> _data;
};