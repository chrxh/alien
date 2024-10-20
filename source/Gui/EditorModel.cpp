#include "EditorModel.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"

void EditorModel::init(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;

    clear();
}

SelectionShallowData const& EditorModel::getSelectionShallowData() const
{
    return _selectionShallowData;
}

void EditorModel::update()
{
    _selectionShallowData = _simulationFacade->getSelectionShallowData();
}

bool EditorModel::isSelectionEmpty() const
{
    return 0 == _selectionShallowData.numCells && 0 == _selectionShallowData.numClusterCells
        && 0 == _selectionShallowData.numParticles;
}

bool EditorModel::isCellSelectionEmpty() const
{
    return 0 == _selectionShallowData.numCells;
}

void EditorModel::clear()
{
    _selectionShallowData = SelectionShallowData();
}

bool EditorModel::existsInspectedEntity(uint64_t id) const
{
    return _inspectedEntityById.find(id) != _inspectedEntityById.end();
}

CellOrParticleDescription EditorModel::getInspectedEntity(uint64_t id) const
{
    return _inspectedEntityById.at(id);
}

void EditorModel::addInspectedEntity(CellOrParticleDescription const& entity)
{
    _inspectedEntityById.emplace(DescriptionEditService::getId(entity), entity);
}

void EditorModel::setInspectedEntities(std::vector<CellOrParticleDescription> const& inspectedEntities)
{
    _inspectedEntityById.clear();
    for (auto const& entity : inspectedEntities) {
        _inspectedEntityById.emplace(DescriptionEditService::getId(entity), entity);
    }
}

bool EditorModel::areEntitiesInspected() const
{
    return !_inspectedEntityById.empty();
}

void EditorModel::setPencilWidth(float value)
{
    _pencilWidth = value;
}

float EditorModel::getPencilWidth() const
{
    return _pencilWidth;
}

void EditorModel::setDefaultColorCode(int value)
{
    _defaultColorCode = value;
}

int EditorModel::getDefaultColorCode() const
{
    return _defaultColorCode;
}

void EditorModel::setForceNoRollout(bool value)
{
    _forceNoRollout = value;
}

void EditorModel::setRolloutToClusters(bool value)
{
    if (!_forceNoRollout) {
        _rolloutToClusters = value;
    }
}

bool EditorModel::isRolloutToClusters() const
{
    return _rolloutToClusters && !_forceNoRollout;
}
