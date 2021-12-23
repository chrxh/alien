#include "EditorModel.h"

#include "Base/Math.h"
#include "EngineImpl/SimulationController.h"
#include "Viewport.h"

namespace
{
    auto const MaxInspectors = 10;
}

_EditorModel::_EditorModel(SimulationController const& simController, Viewport const& viewport)
    : _simController(simController)
    , _viewport(viewport)
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

std::vector<InspectorWindow>& _EditorModel::getInspectorWindows()
{
    return _inspectorWindows;
}

namespace
{
    uint64_t getId(CellOrParticleDescription const& entity)
    {
        if (std::holds_alternative<CellDescription>(entity)) {
            return std::get<CellDescription>(entity).id;
        }
        return std::get<ParticleDescription>(entity).id;
    }

    RealVector2D getPos(CellOrParticleDescription const& entity)
    {
        if (std::holds_alternative<CellDescription>(entity)) {
            return std::get<CellDescription>(entity).pos;
        }
        return std::get<ParticleDescription>(entity).pos;
    }

}

bool _EditorModel::inspectEntities(std::vector<CellOrParticleDescription> const& entities)
{
    std::set<uint64_t> inspectedIds;
    for (auto const& inspectorWindow : _inspectorWindows) {
        inspectedIds.insert(getId(inspectorWindow->getDescription()));
    }
    auto origInspectedIds = inspectedIds;
    for (auto const& entity : entities) {
        inspectedIds.insert(getId(entity));
    }
    if (inspectedIds.size() > MaxInspectors) {
        return false;
    }

    std::vector<CellOrParticleDescription> newEntities;
    for (auto const& entity : entities) {
        if (origInspectedIds.find(getId(entity)) == origInspectedIds.end()) {
            newEntities.emplace_back(entity);
        }
    }
    if (newEntities.empty()) {
        return false;
    }
    RealVector2D center;
    int num = 0;
    for (auto const& entity : newEntities) {
        auto entityPos = _viewport->mapWorldToViewPosition(getPos(entity));
        center += entityPos;
        ++num;
    }
    center = center / num;

    float maxDistanceFromCenter = 0;
    for (auto const& entity : newEntities) {
        auto entityPos = _viewport->mapWorldToViewPosition(getPos(entity));
        auto distanceFromCenter = toFloat(Math::length(entityPos - center));
        maxDistanceFromCenter = std::max(maxDistanceFromCenter, distanceFromCenter);
    }
    auto viewSize = _viewport->getViewSize();
    auto viewRadius = std::min(viewSize.x, viewSize.y) / 2;
    auto factor = maxDistanceFromCenter == 0 ? 1.0f : viewRadius / maxDistanceFromCenter / 1.2f;

    for (auto const& entity : newEntities) {
        auto entityPos = _viewport->mapWorldToViewPosition(getPos(entity));
        auto windowPos = (entityPos - center) * factor + center;
        windowPos.x = std::min(std::max(windowPos.x, 0.0f), toFloat(viewSize.x) - 100.0f);
        windowPos.y = std::min(std::max(windowPos.y, 0.0f), toFloat(viewSize.y) - 100.0f);
        _inspectorWindows.emplace_back(boost::make_shared<_InspectorWindow>(entity, windowPos));
    }
    return true;
}
