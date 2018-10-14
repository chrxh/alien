#include "ModelGpuData.h"

ModelGpuData::ModelGpuData(map<string, int> const & data)
	: _data(data)
{
}

ModelGpuData::ModelGpuData()
{
}

map<string, int> ModelGpuData::getData() const
{
	return _data;
}
