#include "EditorModel.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"

_EditorModel::_EditorModel(SimulationFacade const& simulationFacade)
    : _simulationFacade(simulationFacade)
{
    clear();
}

SelectionShallowData const& _EditorModel::getSelectionShallowData() const
{
    return _selectionShallowData;
}

void _EditorModel::update()
{
    _selectionShallowData = _simulationFacade->getSelectionShallowData();
}

bool _EditorModel::isSelectionEmpty() const
{
    return 0 == _selectionShallowData.numCells && 0 == _selectionShallowData.numClusterCells
        && 0 == _selectionShallowData.numParticles;
}

bool _EditorModel::isCellSelectionEmpty() const
{
    return 0 == _selectionShallowData.numCells;
}

void _EditorModel::clear()
{
    _selectionShallowData = SelectionShallowData();
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
    _inspectedEntityById.emplace(DescriptionEditService::getId(entity), entity);
}

void _EditorModel::setInspectedEntities(std::vector<CellOrParticleDescription> const& inspectedEntities)
{
    _inspectedEntityById.clear();
    for (auto const& entity : inspectedEntities) {
        _inspectedEntityById.emplace(DescriptionEditService::getId(entity), entity);
    }
}

bool _EditorModel::areEntitiesInspected() const
{
    return !_inspectedEntityById.empty();
}

void _EditorModel::setPencilWidth(float value)
{
    _pencilWidth = value;
}

float _EditorModel::getPencilWidth() const
{
    return _pencilWidth;
}

void _EditorModel::setDefaultColorCode(int value)
{
    _defaultColorCode = value;
}

int _EditorModel::getDefaultColorCode() const
{
    return _defaultColorCode;
}

void _EditorModel::setForceNoRollout(bool value)
{
    _forceNoRollout = value;
}

void _EditorModel::setRolloutToClusters(bool value)
{
    if (!_forceNoRollout) {
        _rolloutToClusters = value;
    }
}

bool _EditorModel::isRolloutToClusters() const
{
    return _rolloutToClusters && !_forceNoRollout;
}
