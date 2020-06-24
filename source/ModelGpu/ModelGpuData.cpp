#include "ModelGpuData.h"

#include "CudaConstants.h"

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


ModelGpuData::ModelGpuData(map<string, int> const & data)
	: _data(data)
{
}

ModelGpuData::ModelGpuData()
{
}

CudaConstants ModelGpuData::getCudaConstants() const
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

int ModelGpuData::getNumThreadsPerBlock()
{
    return _data.at(numThreadsPerBlock_key);
}

void ModelGpuData::setNumThreadsPerBlock(int value)
{
    _data.insert_or_assign(numThreadsPerBlock_key, value);
}

int ModelGpuData::getNumBlocks()
{
    return _data.at(numBlocks_key);
}

void ModelGpuData::setNumBlocks(int value)
{
    _data.insert_or_assign(numBlocks_key, value);
}

int ModelGpuData::getMaxClusters()
{
    return _data.at(maxClusters_key);
}

void ModelGpuData::setMaxClusters(int value)
{
    _data.insert_or_assign(maxClusters_key, value);
}

int ModelGpuData::getMaxCells()
{
    return _data.at(maxCells_key);
}

void ModelGpuData::setMaxCells(int value)
{
    _data.insert_or_assign(maxCells_key, value);
}

int ModelGpuData::getMaxParticles()
{
    return _data.at(maxParticles_key);
}

void ModelGpuData::setMaxParticles(int value)
{
    _data.insert_or_assign(maxParticles_key, value);
}

int ModelGpuData::getMaxTokens()
{
    return _data.at(maxTokens_key);
}

void ModelGpuData::setMaxTokens(int value)
{
    _data.insert_or_assign(maxTokens_key, value);
}

int ModelGpuData::getDynamicMemorySize()
{
    return _data.at(dynamicMemorySize_key);
}

void ModelGpuData::setDynamicMemorySize(int value)
{
    _data.insert_or_assign(dynamicMemorySize_key, value);
}

void ModelGpuData::setMaxCellPointers(int value)
{
    _data.insert_or_assign(maxCellPointers_key, value);
}

void ModelGpuData::setMaxClusterPointers(int value)
{
    _data.insert_or_assign(maxClusterPointers_key, value);
}

void ModelGpuData::setMaxParticlePointers(int value)
{
    _data.insert_or_assign(maxParticlePointers_key, value);
}

void ModelGpuData::setMaxTokenPointers(int value)
{
    _data.insert_or_assign(maxTokenPointers_key, value);
}

int ModelGpuData::getMetadataDynamicMemorySize()
{
    return _data.at(metadataDynamicMemorySize_key);
}

void ModelGpuData::setMetadataDynamicMemorySize(int value)
{
    _data.insert_or_assign(metadataDynamicMemorySize_key, value);
}

map<string, int> ModelGpuData::getData() const
{
    return _data;
}
