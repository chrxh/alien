#pragma once

#include "Definitions.h"
#include "DefinitionsImpl.h"

class MODELGPU_EXPORT ModelGpuData
{
public:
    ModelGpuData() = default;
    explicit ModelGpuData(map<string, int> const& data);
	explicit ModelGpuData(CudaConstants const& value);

    CudaConstants getCudaConstants() const;

    map<string, int> getData() const;

private:
	map<string, int> _data;
};