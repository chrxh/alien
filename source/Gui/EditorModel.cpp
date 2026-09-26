#include "EditorModel.h"

#include <Data/DescEditService.h>

#include <EngineInterface/SimulationFacade.h>

#include <EngineInterface/SimulationFacade.h>

void EditorModel::setup()
{
    clear();
}

SelectionShallowData const& EditorModel::getSelectionShallowData() const
{
    return _selectionShallowData;
}

void EditorModel::update()
{
    _selectionShallowData = _SimulationFacade::get()->getSelectionShallowData();
}

bool EditorModel::isSelectionEmpty() const
{
    return 0 == _selectionShallowData.numObjects && 0 == _selectionShallowData.numClusterCells && 0 == _selectionShallowData.numEnergyParticles;
}

bool EditorModel::isCellSelectionEmpty() const
{
    return 0 == _selectionShallowData.numObjects;
}

void EditorModel::clear()
{
    _selectionShallowData = SelectionShallowData();
}

bool EditorModel::existsInspectedEntity(uint64_t id) const
{
    return _inspectedEntityById.find(id) != _inspectedEntityById.end();
}

ExtendedObjectOrEnergyDesc EditorModel::getInspectedEntity(uint64_t id) const
{
    return _inspectedEntityById.at(id);
}

void EditorModel::addInspectedEntity(ExtendedObjectOrEnergyDesc const& entity)
{
    _inspectedEntityById.insert_or_assign(DescEditService::get().getId(entity), entity);
}

void EditorModel::setInspectedEntities(std::vector<ExtendedObjectOrEnergyDesc> const& inspectedEntities)
{
    _inspectedEntityById.clear();
    for (auto const& entity : inspectedEntities) {
        _inspectedEntityById.emplace(DescEditService::get().getId(entity), entity);
    }
}

bool EditorModel::areEntitiesInspected() const
{
    return !_inspectedEntityById.empty();
}

void EditorModel::setDefaultColorCode(int value)
{
    _defaultColorCode = value;
}

int EditorModel::getDefaultColorCode() const
{
    return _defaultColorCode;
}

EditTool EditorModel::getTool() const
{
    return _tool;
}

void EditorModel::setTool(EditTool value)
{
    _tool = value;
}

bool EditorModel::isApplyToNetworks() const
{
    return _applyToNetworks != _scopeInvertedTemporarily;
}

bool EditorModel::isApplyToNetworksPersistent() const
{
    return _applyToNetworks;
}

void EditorModel::setApplyToNetworks(bool value)
{
    _applyToNetworks = value;
}

void EditorModel::setScopeInvertedTemporarily(bool value)
{
    _scopeInvertedTemporarily = value;
}

bool EditorModel::isGlueOnContact() const
{
    return _glueOnContact;
}

void EditorModel::setGlueOnContact(bool value)
{
    _glueOnContact = value;
}

SelectionBounds EditorModel::getSelectionBounds(bool includeClusters) const
{
    auto const& data = _selectionShallowData;
    if (includeClusters) {
        return SelectionBounds{
            .center = {data.clusterCenterPosX, data.clusterCenterPosY},
            .velocity = {data.clusterCenterVelX, data.clusterCenterVelY},
            .topLeft = {data.clusterMinPosX, data.clusterMinPosY},
            .bottomRight = {data.clusterMaxPosX, data.clusterMaxPosY},
        };
    }
    return SelectionBounds{
        .center = {data.centerPosX, data.centerPosY},
        .velocity = {data.centerVelX, data.centerVelY},
        .topLeft = {data.minPosX, data.minPosY},
        .bottomRight = {data.maxPosX, data.maxPosY},
    };
}
