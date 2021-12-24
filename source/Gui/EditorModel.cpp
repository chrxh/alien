#include "EditorModel.h"

#include "EngineImpl/SimulationController.h"

_EditorModel::_EditorModel(SimulationController const& simController)
    : _simController(simController)
{
    clear();
}

SelectionShallowData const& _EditorModel::getSelectionShallowData() const
{
    return _selectionShallowData;
}

void _EditorModel::update()
{
    _selectionShallowData = _simController->getSelectionShallowData();
}

bool _EditorModel::isSelectionEmpty() const
{
    return 0 == _selectionShallowData.numCells && 0 == _selectionShallowData.numClusterCells
        && 0 == _selectionShallowData.numParticles;
}

void _EditorModel::clear()
{
    _selectionShallowData = SelectionShallowData();
}

std::vector<CellOrParticleDescription> _EditorModel::fetchEntitiesToInspect()
{
    auto result = _entitiesToInspect;
    _entitiesToInspect = {};
    return result;
}

void _EditorModel::inspectEntities(std::vector<CellOrParticleDescription> const& entities)
{
    _entitiesToInspect = entities;
}
