#include "GraphicsItemConfig.h"

void GraphicsItemConfig::init(SimulationParameters * parameters)
{
	_parameters = parameters;
}

bool GraphicsItemConfig::isShowCellInfo() const
{
	return _showCellInfo;
}

SimulationParameters * GraphicsItemConfig::getSimulationParameters() const
{
	return _parameters;
}
