#pragma once

#include "Definitions.h"
#include "DefinitionsImpl.h"

class MODELGPU_EXPORT ModelGpuData
{
public:
	ModelGpuData(map<string, int> const& data);
	ModelGpuData();

    CudaConstants getCudaConstants() const;

    int getNumThreadsPerBlock();
    void setNumThreadsPerBlock(int value);

    int getNumBlocks();
    void setNumBlocks(int value);

    int getMaxClusters();
    void setMaxClusters(int value);

    int getMaxCells();
    void setMaxCells(int value);

    int getMaxParticles();
    void setMaxParticles(int value);

    int getMaxTokens();
    void setMaxTokens(int value);

    int getDynamicMemorySize();
    void setDynamicMemorySize(int value);

    void setMaxCellPointers(int value);
    void setMaxClusterPointers(int value);
    void setMaxParticlePointers(int value);
    void setMaxTokenPointers(int value);

    int getMetadataDynamicMemorySize();
    void setMetadataDynamicMemorySize(int value);

	map<string, int> getData() const;

private:
	map<string, int> _data;
};