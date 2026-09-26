#include "InspectionController.h"

#include <unordered_map>
#include <unordered_set>

#include <Base/Math.h>

#include <Data/CellTypeConstants.h>
#include <Data/DescEditService.h>

#include <EngineInterface/InspectedEntityIds.h>
#include <EngineInterface/SimulationFacade.h>

#include "EditorController.h"
#include "EditorModel.h"
#include "GenericMessageDialog.h"
#include "GenomeEditorWindow.h"
#include "InspectionWindow.h"
#include "OverlayController.h"
#include "Viewport.h"

namespace
{
    auto constexpr MaxInspectedGenomes = 20;
}

bool InspectionController::areInspectionWindowsActive() const
{
    return !_inspectorWindows.empty();
}

void InspectionController::onCloseAllInspectorWindows()
{
    _inspectorWindows.clear();
}

bool InspectionController::isObjectInspectionPossible() const
{
    return !EditorModel::get().isSelectionEmpty();
}

bool InspectionController::isGenomeInspectionPossible() const
{
    return EditorModel::get().getSelectionShallowData().numCreatures > 0;
}

bool InspectionController::isCreatureInspectionPossible() const
{
    return EditorModel::get().getSelectionShallowData().numCreatures > 0;
}

void InspectionController::onInspectSelectedObjects()
{
    ContentDesc selectedData = _SimulationFacade::get()->getSelectedSimulationData(false);
    if (!inspectObjects(DescEditService::get().getObjects(selectedData), false)) {
        std::string message = "Too many objects are selected for inspection. A maximum of ";
        message += std::to_string(Const::MaxInspectedObjects);
        message += " objects are allowed.";
        showMessage("Inspection not possible", message);
    }
}

void InspectionController::onInspectSelectedGenomes()
{
    ContentDesc selectedData = _SimulationFacade::get()->getSelectedSimulationData(false);

    // A genome can be carried by creatures of different lineages; only an unambiguous lineage marks the tab
    std::map<uint64_t, std::optional<int>> lineageByGenomeId;
    for (auto const& creature : selectedData._creatures) {
        auto [entry, inserted] = lineageByGenomeId.try_emplace(creature._genomeId, creature._lineageId);
        if (!inserted && entry->second != creature._lineageId) {
            entry->second.reset();
        }
    }

    struct InspectedGenome
    {
        GenomeDesc genome;
        std::optional<int> lineageId;
    };
    std::vector<InspectedGenome> uniqueGenomes;
    for (auto const& genome : selectedData._genomes) {
        auto lineageId = lineageByGenomeId.contains(genome._id) ? lineageByGenomeId.at(genome._id) : std::nullopt;
        auto genomeWithoutId = genome;
        genomeWithoutId._id = 0;

        InspectedGenome* collected = nullptr;
        for (auto& existing : uniqueGenomes) {
            if (existing.genome == genomeWithoutId) {
                collected = &existing;
                break;
            }
        }
        if (collected != nullptr) {
            if (collected->lineageId != lineageId) {
                collected->lineageId.reset();
            }
        } else {
            uniqueGenomes.emplace_back(genomeWithoutId, lineageId);
        }
    }

    if (uniqueGenomes.size() > MaxInspectedGenomes) {
        std::string message = "Too many genomes are selected for inspection. A maximum of ";
        message += std::to_string(MaxInspectedGenomes);
        message += " genomes are allowed.";
        showMessage("Inspection not possible", message);
        return;
    }

    for (auto const& uniqueGenome : uniqueGenomes) {
        GenomeEditorWindow::get().openTab(uniqueGenome.genome, false, true, uniqueGenome.lineageId);
    }

    printOverlayMessage(std::to_string(uniqueGenomes.size()) + (uniqueGenomes.size() == 1 ? " genome" : " genomes") + " inspected");
}

void InspectionController::onInspectSelectedCreatures()
{
    // Collect the creature ids of the directly selected cells
    ContentDesc directlySelectedData = _SimulationFacade::get()->getSelectedSimulationData(false);
    std::unordered_set<uint64_t> selectedCreatureIds;
    for (auto const& object : directlySelectedData._objects) {
        if (object.getObjectType() == ObjectType_Cell) {
            selectedCreatureIds.insert(object.getCellRef()._creatureId);
        }
    }
    if (selectedCreatureIds.empty()) {
        return;
    }

    // Consider all cells of those creatures and pick the head cell of each
    // (head cell with the smallest branch index, then smallest concatenation index)
    ContentDesc creatureData = _SimulationFacade::get()->getSelectedSimulationData(true);
    auto entities = DescEditService::get().getObjects(creatureData);

    std::unordered_map<uint64_t, ExtendedObjectDesc> headCellByCreatureId;
    for (auto const& entity : entities) {
        if (!std::holds_alternative<ExtendedObjectDesc>(entity)) {
            continue;
        }
        auto const& extendedObject = std::get<ExtendedObjectDesc>(entity);
        if (extendedObject.object.getObjectType() != ObjectType_Cell || !extendedObject.creature.has_value()) {
            continue;
        }
        auto const& cell = extendedObject.object.getCellRef();
        if (!cell._headCell || selectedCreatureIds.find(cell._creatureId) == selectedCreatureIds.end()) {
            continue;
        }
        auto it = headCellByCreatureId.find(cell._creatureId);
        if (it == headCellByCreatureId.end()) {
            headCellByCreatureId.emplace(cell._creatureId, extendedObject);
        } else {
            auto const& bestCell = it->second.object.getCellRef();
            if (cell._branchIndex < bestCell._branchIndex
                || (cell._branchIndex == bestCell._branchIndex && cell._concatenationIndex < bestCell._concatenationIndex)) {
                it->second = extendedObject;
            }
        }
    }

    std::vector<ExtendedObjectOrEnergyDesc> headCells;
    for (auto const& [creatureId, headCell] : headCellByCreatureId) {
        headCells.emplace_back(headCell);
    }

    if (!inspectObjects(headCells, true)) {
        std::string message = "Too many creatures are selected for inspection. A maximum of ";
        message += std::to_string(Const::MaxInspectedObjects);
        message += " creatures are allowed.";
        showMessage("Inspection not possible", message);
    }
}

void InspectionController::process()
{
    if (!EditorController::get().isOn()) {
        return;
    }

    // Process inspector windows
    for (auto const& inspectorWindow : _inspectorWindows) {
        inspectorWindow->process();
    }

    // Inspector windows closed?
    std::vector<InspectionWindow> inspectorWindows;
    std::vector<ExtendedObjectOrEnergyDesc> inspectedEntities;
    for (auto const& inspectorWindow : _inspectorWindows) {
        if (!inspectorWindow->isClosed()) {
            inspectorWindows.emplace_back(inspectorWindow);

            auto id = inspectorWindow->getId();
            inspectedEntities.emplace_back(EditorModel::get().getInspectedEntity(id));
        }
    }
    _inspectorWindows = inspectorWindows;
    EditorModel::get().setInspectedEntities(inspectedEntities);
    if (inspectedEntities.empty()) {
        return;
    }

    // Get inspected entities from simulation periodically
    static int counter = 0;
    if (++counter == 10) {
        std::vector<uint64_t> entityIds;
        for (auto const& entity : inspectedEntities) {
            entityIds.emplace_back(DescEditService::get().getId(entity));
        }
        auto inspectedData = _SimulationFacade::get()->getInspectedSimulationData(entityIds);
        auto newInspectedEntities = DescEditService::get().getObjects(inspectedData);
        EditorModel::get().setInspectedEntities(newInspectedEntities);
        counter = 0;
    }

    inspectorWindows.clear();
    for (auto const& inspectorWindow : _inspectorWindows) {
        if (EditorModel::get().existsInspectedEntity(inspectorWindow->getId())) {
            inspectorWindows.emplace_back(inspectorWindow);
        }
    }
    _inspectorWindows = inspectorWindows;
}

bool InspectionController::inspectObjects(std::vector<ExtendedObjectOrEnergyDesc> const& entities, bool creatureMode)
{
    if (entities.empty()) {
        return true;
    }

    // Filter entities if cells are selected
    std::vector<ExtendedObjectOrEnergyDesc> filteredEntities;
    auto areCellsSelected = false;
    for (auto const& cellOrParticle : entities) {
        if (std::holds_alternative<ExtendedObjectDesc>(cellOrParticle)) {
            areCellsSelected = true;
            break;
        }
    }
    if (areCellsSelected) {
        for (auto const& cellOrParticle : entities) {
            if (std::holds_alternative<ExtendedObjectDesc>(cellOrParticle)) {
                filteredEntities.emplace_back(cellOrParticle);
            }
        }
    } else {
        filteredEntities = entities;
    }

    auto borderlessRendering = _SimulationFacade::get()->getSimulationParameters().borderlessRendering.value;

    std::set<uint64_t> inspectedIds;
    for (auto const& inspectorWindow : _inspectorWindows) {
        inspectedIds.insert(inspectorWindow->getId());
    }
    auto origInspectedIds = inspectedIds;
    for (auto const& entity : filteredEntities) {
        inspectedIds.insert(DescEditService::get().getId(entity));
    }

    std::vector<ExtendedObjectOrEnergyDesc> newEntities;
    for (auto const& entity : filteredEntities) {
        if (origInspectedIds.find(DescEditService::get().getId(entity)) == origInspectedIds.end()) {
            newEntities.emplace_back(entity);
        }
    }
    if (newEntities.empty()) {
        return true;
    }
    if (inspectedIds.size() > Const::MaxInspectedObjects) {
        return false;
    }
    RealVector2D center;
    int num = 0;
    for (auto const& entity : filteredEntities) {
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescEditService::get().getPos(entity), borderlessRendering);
        center += entityPos;
        ++num;
    }
    center = center / num;

    float maxDistanceFromCenter = 0;
    for (auto const& entity : filteredEntities) {
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescEditService::get().getPos(entity), borderlessRendering);
        auto distanceFromCenter = toFloat(Math::length(entityPos - center));
        maxDistanceFromCenter = std::max(maxDistanceFromCenter, distanceFromCenter);
    }
    auto viewSize = Viewport::get().getViewSize();
    auto factorX = maxDistanceFromCenter == 0 ? 1.0f : viewSize.x / maxDistanceFromCenter / 3.8f;
    auto factorY = maxDistanceFromCenter == 0 ? 1.0f : viewSize.y / maxDistanceFromCenter / 3.4f;

    for (auto const& entity : newEntities) {
        auto id = DescEditService::get().getId(entity);
        EditorModel::get().addInspectedEntity(entity);
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescEditService::get().getPos(entity), borderlessRendering);
        auto windowPosX = (entityPos.x - center.x) * factorX + center.x;
        auto windowPosY = (entityPos.y - center.y) * factorY + center.y;
        windowPosX = std::min(std::max(windowPosX, 0.0f), toFloat(viewSize.x) - 300.0f) + 40.0f;
        windowPosY = std::min(std::max(windowPosY, 0.0f), toFloat(viewSize.y) - 500.0f) + 40.0f;
        _inspectorWindows.emplace_back(std::make_shared<_InspectionWindow>(id, RealVector2D{windowPosX, windowPosY}, creatureMode));
    }
    return true;
}
