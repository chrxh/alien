#include "ItemConfig.h"

void ItemConfig::init(SimulationParameters const* parameters)
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

SimulationParameters const* ItemConfig::getSimulationParameters() const
{
	return _parameters;
}
