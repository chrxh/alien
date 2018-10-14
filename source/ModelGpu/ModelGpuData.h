#pragma once

#include "Definitions.h"

class MODELGPU_EXPORT ModelGpuData
{
public:
	ModelGpuData(map<string, int> const& data);
	ModelGpuData();

	map<string, int> getData() const;

private:
	map<string, int> _data;
};