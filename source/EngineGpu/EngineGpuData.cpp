#include "EngineGpuData.h"

#include "EngineGpuKernels/CudaConstants.h"

namespace
{
    string const numThreadsPerBlock_key = "numThreadsPerBlock";
    string const numBlocks_key = "numBlocks";

    string const maxClusters_key = "maxClusters";
    string const maxCells_key = "maxCells";
    string const maxParticles_key = "maxParticles";
    string const maxTokens_key = "maxTokens";
    string const maxCellPointers_key = "maxCellPointers";
    string const maxClusterPointers_key = "maxClusterPointers";
    string const maxParticlePointers_key = "maxParticlePointers";
    string const maxTokenPointers_key = "maxTokenPointers";
    string const dynamicMemorySize_key = "dynamicMemorySize";
    string const metadataDynamicMemorySize_key = "stringByteSize";
}


EngineGpuData::EngineGpuData(map<string, int> const & data)
	: _data(data)
{
}

EngineGpuData::EngineGpuData(CudaConstants const & value)
{
    _data.insert_or_assign(numThreadsPerBlock_key, value.NUM_THREADS_PER_BLOCK);
    _data.insert_or_assign(numBlocks_key, value.NUM_BLOCKS);
    _data.insert_or_assign(maxClusters_key, value.MAX_CLUSTERS);
    _data.insert_or_assign(maxClusterPointers_key, value.MAX_CLUSTERPOINTERS);
    _data.insert_or_assign(maxCells_key, value.MAX_CELLS);
    _data.insert_or_assign(maxCellPointers_key, value.MAX_CELLPOINTERS);
    _data.insert_or_assign(maxParticles_key, value.MAX_PARTICLES);
    _data.insert_or_assign(maxParticlePointers_key, value.MAX_PARTICLEPOINTERS);
    _data.insert_or_assign(maxTokens_key, value.MAX_TOKENS);
    _data.insert_or_assign(maxTokenPointers_key, value.MAX_TOKENPOINTERS);
    _data.insert_or_assign(dynamicMemorySize_key, value.DYNAMIC_MEMORY_SIZE);
    _data.insert_or_assign(metadataDynamicMemorySize_key, value.METADATA_DYNAMIC_MEMORY_SIZE);
}

CudaConstants EngineGpuData::getCudaConstants() const
{
    CudaConstants result;
    result.NUM_THREADS_PER_BLOCK = _data.at(numThreadsPerBlock_key);
    result.NUM_BLOCKS = _data.at(numBlocks_key);
    result.MAX_CLUSTERS = _data.at(maxClusters_key);
    result.MAX_CELLS = _data.at(maxCells_key);
    result.MAX_PARTICLES = _data.at(maxParticles_key);
    result.MAX_TOKENS = _data.at(maxTokens_key);
    result.MAX_CELLPOINTERS = _data.at(maxCellPointers_key);
    result.MAX_CLUSTERPOINTERS = _data.at(maxClusterPointers_key);
    result.MAX_PARTICLEPOINTERS = _data.at(maxParticlePointers_key);
    result.MAX_TOKENPOINTERS = _data.at(maxTokenPointers_key);
    result.DYNAMIC_MEMORY_SIZE = _data.at(dynamicMemorySize_key);
    result.METADATA_DYNAMIC_MEMORY_SIZE = _data.at(metadataDynamicMemorySize_key);
    return result;
}

map<string, int> EngineGpuData::getData() const
{
    return _data;
}
