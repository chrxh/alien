#include "EditorModel.h"

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"

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

bool _EditorModel::existsInspectedEntity(uint64_t id) const
{
    return _inspectedEntityById.find(id) != _inspectedEntityById.end();
}

CellOrParticleDescription _EditorModel::getInspectedEntity(uint64_t id) const
{
    return _inspectedEntityById.at(id);
}

void _EditorModel::addInspectedEntity(CellOrParticleDescription const& entity)
{
    _inspectedEntityById.emplace(DescriptionHelper::getId(entity), entity);
}

void _EditorModel::setInspectedEntities(std::vector<CellOrParticleDescription> const& inspectedEntities)
{
    _inspectedEntityById.clear();
    for (auto const& entity : inspectedEntities) {
        _inspectedEntityById.emplace(DescriptionHelper::getId(entity), entity);
    }
}

void _EditorModel::setDrawMode(bool value)
{
    _drawMode = value;
}

bool _EditorModel::isDrawMode() const
{
    return _drawMode;
}
