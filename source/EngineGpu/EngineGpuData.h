#pragma once

#include "Definitions.h"
#include "DefinitionsImpl.h"

class ENGINEGPU_EXPORT EngineGpuData
{
public:
    EngineGpuData() = default;
    explicit EngineGpuData(map<string, int> const& data);
	explicit EngineGpuData(CudaConstants const& value);

    CudaConstants getCudaConstants() const;

    map<string, int> getData() const;

private:
	map<string, int> _data;
};