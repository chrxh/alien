#pragma once

#include "Definitions.h"
#include "DefinitionsImpl.h"

class MODELGPU_EXPORT ModelGpuData
{
public:
	ModelGpuData(map<string, int> const& data);
	ModelGpuData();

    CudaConstants getCudaConstants() const;

    void setNumThreadsPerBlock(int value);
    void setNumBlocks(int value);

    void setNumClusterPointerArrays(int value);
    void setMaxClusters(int value);
    void setMaxCells(int value);
    void setMaxParticles(int value);
    void setMaxTokens(int value);
    void setMaxCellPointers(int value);
    void setMaxClusterPointers(int value);
    void setMaxParticlePointers(int value);
    void setMaxTokenPointers(int value);
    void setDynamicMemorySize(int value);

	map<string, int> getData() const;

private:
	map<string, int> _data;
};