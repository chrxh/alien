#include "ModelCpuData.h"

namespace
{
	string const maxRunningThreads_key = "maxRunningThreads";
	string const gridSizeX_key = "gridSize.x";
	string const gridSizeY_key = "gridSize.y";
}

ModelCpuData::ModelCpuData(map<string, int> const & data)
	: _data(data)
{
}

ModelCpuData::ModelCpuData(uint maxRunningThreads, IntVector2D const & gridSize)
{
	_data.insert_or_assign(maxRunningThreads_key, static_cast<int>(maxRunningThreads));
	_data.insert_or_assign(gridSizeX_key, gridSize.x);
	_data.insert_or_assign(gridSizeY_key, gridSize.y);
}

uint ModelCpuData::getMaxRunningThreads() const
{
	return static_cast<uint>(_data.at(maxRunningThreads_key));
}

IntVector2D ModelCpuData::getGridSize() const
{
	return{ _data.at(gridSizeX_key), _data.at(gridSizeY_key) };
}

map<string, int> ModelCpuData::getData() const
{
	return _data;
}
