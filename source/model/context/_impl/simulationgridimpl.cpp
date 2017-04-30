#include "simulationgridimpl.h"

SimulationGridImpl::SimulationGridImpl(QObject * parent)
	: SimulationGrid(parent)
{
}

void SimulationGridImpl::init(IntVector2D gridSize)
{
	for (int x = 0; x < gridSize.x; ++x) {
		_units.push_back(std::vector<SimulationUnit*>(gridSize.y, nullptr));
	}
}

void SimulationGridImpl::deleteUnits()
{
	for (auto const& unitCol : _units) {
		for (auto const& unit : unitCol) {
			delete unit;
		}
	}
}

void SimulationGridImpl::registerUnit(IntVector2D gridPos, SimulationUnit * unit)
{
	_units[gridPos.x][gridPos.y] = unit;
}
