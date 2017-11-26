#include "ItemConfig.h"

void ItemConfig::init(SimulationParameters * parameters)
{
	_parameters = parameters;
}

bool ItemConfig::isShowCellInfo() const
{
	return _showCellInfo;
}

void ItemConfig::setShowCellInfo(bool value)
{
	_showCellInfo = value;
}

SimulationParameters * ItemConfig::getSimulationParameters() const
{
	return _parameters;
}
