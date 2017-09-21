#include "ItemConfigT.h"

void ItemConfig::init(SimulationParameters * parameters)
{
	_parameters = parameters;
}

bool ItemConfig::isShowCellInfo() const
{
	return _showCellInfo;
}

SimulationParameters * ItemConfig::getSimulationParameters() const
{
	return _parameters;
}
