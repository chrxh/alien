#include "CreatorController.h"

#include <algorithm>

#include <imgui.h>

#include <Base/GlobalSettings.h>
#include <Base/Math.h>

#include <EngineInterface/SimulationFacade.h>

#include "SimulationInteractionController.h"
#include "StyleService.h"
#include "Viewport.h"

namespace
{
    auto constexpr PreviewLineThickness = 2.0f;
    auto constexpr PreviewPointRadius = 3.0f;
    auto constexpr PlacementPreviewMinRadius = 1.5f;
    auto constexpr PlacementPreviewRadiusFactor = 0.3f;
    auto constexpr MaxNumPlacementPreviewPoints = 10000;

    ImVec2 mapToViewPosition(RealVector2D const& worldPos)
    {
        auto viewPos = Viewport::get().mapWorldToViewPosition(worldPos);
        return ImVec2(viewPos.x, viewPos.y);
    }

    std::vector<ImVec2> mapToViewPositions(std::vector<RealVector2D> const& worldPositions)
    {
        std::vector<ImVec2> result;
        result.reserve(worldPositions.size());
        for (auto const& worldPos : worldPositions) {
            result.emplace_back(mapToViewPosition(worldPos));
        }
        return result;
    }

    bool isPlacementTool(EditTool tool)
    {
        return tool == EditTool_Object || tool == EditTool_Rectangle || tool == EditTool_Hexagon || tool == EditTool_Disc;
    }

    bool isPointTool(EditTool tool)
    {
        return tool == EditTool_Line || tool == EditTool_Curve || tool == EditTool_Polygon;
    }
}

CreatorParameters const& CreatorController::getParameters() const
{
    return _parameters;
}

void CreatorController::setParameters(CreatorParameters const& parameters)
{
    _parameters = parameters;
    validateAndCorrect();
}

bool CreatorController::isFinishingPointsPossible() const
{
    return toInt(_points.size()) >= getMinNumPoints();
}

void CreatorController::onFinishPoints()
{
    if (!isFinishingPointsPossible()) {
        return;
    }
    auto tool = EditorModel::get().getTool();
    if (tool == EditTool_Line) {
        addToSimulation(CreatorService::get().createLine(getObjectProperties(), _points, _parameters.objectDistance));
    } else if (tool == EditTool_Curve) {
        addToSimulation(CreatorService::get().createCurve(getObjectProperties(), _points, _parameters.objectDistance));
    } else if (tool == EditTool_Polygon) {
        addToSimulation(CreatorService::get().createPolygon(getObjectProperties(), _points, _parameters.objectDistance));
    }
    _points.clear();
    EditorModel::get().update();
}

bool CreatorController::hasPoints() const
{
    return !_points.empty();
}

void CreatorController::onAbortPoints()
{
    _points.clear();
}

void CreatorController::onLeftMouseButtonPressed(RealVector2D const& viewPos)
{
    auto tool = EditorModel::get().getTool();
    auto worldPos = Viewport::get().mapViewToWorldPosition(viewPos);
    if (isPlacementTool(tool)) {
        place(worldPos);
    } else if (isPointTool(tool)) {
        _points.emplace_back(worldPos);
    } else if (tool == EditTool_Freehand) {
        draw(worldPos);
    }
}

void CreatorController::onLeftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    if (EditorModel::get().getTool() == EditTool_Freehand) {
        draw(Viewport::get().mapViewToWorldPosition(viewPos));
    }
}

void CreatorController::onLeftMouseButtonReleased(RealVector2D const& viewPos, RealVector2D const& prevViewPos)
{
    if (EditorModel::get().getTool() == EditTool_Freehand) {
        finishDrawing();
    }
}

void CreatorController::onRightMouseButtonPressed(RealVector2D const& viewPos)
{
    if (isPointTool(EditorModel::get().getTool()) && !_points.empty()) {
        _points.pop_back();
    }
}

bool CreatorController::isCrosshairCursor() const
{
    return EditorModel::get().getTool() != EditTool_Freehand;
}

void CreatorController::drawCursor(ImDrawList* drawList, ImVec2 const& mousePos) const
{
    if (EditorModel::get().getTool() == EditTool_Freehand) {
        auto radius = _parameters.pencilWidth * Viewport::get().getZoomFactor();
        drawList->AddCircleFilled(mousePos, radius, Const::ConstructionPreviewBrushColor);
    }
}

void CreatorController::init()
{
    auto& settings = GlobalSettings::get();
    _parameters.energy = settings.getValue("editors.creator.energy", _parameters.energy);
    _parameters.stiffness = settings.getValue("editors.creator.stiffness", _parameters.stiffness);
    _parameters.isStatic = settings.getValue("editors.creator.static", _parameters.isStatic);
    _parameters.objectDistance = settings.getValue("editors.creator.object distance", _parameters.objectDistance);
    _parameters.glow = settings.getValue("editors.creator.glow", _parameters.glow);
    _parameters.sticky = settings.getValue("editors.creator.make sticky", _parameters.sticky);
    _parameters.rectHorizontalObjects = settings.getValue("editors.creator.rect horizontal objects", _parameters.rectHorizontalObjects);
    _parameters.rectVerticalObjects = settings.getValue("editors.creator.rect vertical objects", _parameters.rectVerticalObjects);
    _parameters.layers = settings.getValue("editors.creator.layers", _parameters.layers);
    _parameters.outerRadius = settings.getValue("editors.creator.outer radius", _parameters.outerRadius);
    _parameters.innerRadius = settings.getValue("editors.creator.inner radius", _parameters.innerRadius);
    _parameters.material = settings.getValue("editors.creator.material", _parameters.material);
    validateAndCorrect();
}

void CreatorController::process()
{
    auto const& interactionController = SimulationInteractionController::get();
    auto tool = interactionController.isEditMode() ? EditorModel::get().getTool() : EditTool_Select;
    if (tool != _lastTool || interactionController.getInteractionMode() == InteractionMode_PositionSelection) {
        onAbortPoints();
        _lastTool = tool;
    }

    switch (tool) {
    case EditTool_Object:
    case EditTool_Rectangle:
    case EditTool_Hexagon:
    case EditTool_Disc:
        processPlacementPreview();
        break;
    case EditTool_Line:
        processPointPreview(_points, false);
        break;
    case EditTool_Curve:
        processControlPolygonPreview();
        processPointPreview(CreatorService::get().calcBezierCurvePath(_points, _parameters.objectDistance), false);
        break;
    case EditTool_Polygon:
        processPointPreview(_points, true);
        break;
    default:
        break;
    }
}

void CreatorController::shutdown()
{
    auto& settings = GlobalSettings::get();
    settings.setValue("editors.creator.energy", _parameters.energy);
    settings.setValue("editors.creator.stiffness", _parameters.stiffness);
    settings.setValue("editors.creator.static", _parameters.isStatic);
    settings.setValue("editors.creator.object distance", _parameters.objectDistance);
    settings.setValue("editors.creator.glow", _parameters.glow);
    settings.setValue("editors.creator.make sticky", _parameters.sticky);
    settings.setValue("editors.creator.rect horizontal objects", _parameters.rectHorizontalObjects);
    settings.setValue("editors.creator.rect vertical objects", _parameters.rectVerticalObjects);
    settings.setValue("editors.creator.layers", _parameters.layers);
    settings.setValue("editors.creator.outer radius", _parameters.outerRadius);
    settings.setValue("editors.creator.inner radius", _parameters.innerRadius);
    settings.setValue("editors.creator.material", _parameters.material);
}

void CreatorController::place(RealVector2D const& worldPos)
{
    addToSimulation(createShape(worldPos));
    EditorModel::get().update();
}

void CreatorController::draw(RealVector2D const& worldPos)
{
    auto createAlignedCircle = [&](RealVector2D const& pos) {
        return CreatorService::get().createPencilDot(getObjectProperties(), pos, _parameters.pencilWidth);
    };

    auto prevEntityCount = isEnergyMaterial() ? _drawingDescription._energies.size() : _drawingDescription._objects.size();

    if (_drawingDescription.isEmpty()) {
        DescEditService::get().addIfSpaceAvailable(
            _drawingDescription, _drawingOccupancy, createAlignedCircle(worldPos), 0.5f, _SimulationFacade::get()->getWorldSize());
        _lastDrawPos = worldPos;
    } else {
        auto posDelta = Math::length(worldPos - _lastDrawPos);
        if (posDelta > 0) {
            auto lastDrawPos = _lastDrawPos;
            for (float interDelta = 0; interDelta < posDelta; interDelta += 1.0f) {
                auto drawPos = lastDrawPos + (worldPos - lastDrawPos) * interDelta / posDelta;
                auto toAdd = createAlignedCircle(drawPos);
                DescEditService::get().addIfSpaceAvailable(_drawingDescription, _drawingOccupancy, toAdd, 0.5f, _SimulationFacade::get()->getWorldSize());
                _lastDrawPos = drawPos;
            }
        }
    }

    auto newEntityCount = isEnergyMaterial() ? _drawingDescription._energies.size() : _drawingDescription._objects.size();
    if (newEntityCount > prevEntityCount) {
        ContentDesc newEntities;
        for (auto i = prevEntityCount; i < newEntityCount; ++i) {
            if (isEnergyMaterial()) {
                newEntities._energies.emplace_back(_drawingDescription._energies.at(i));
            } else {
                newEntities._objects.emplace_back(_drawingDescription._objects.at(i));
            }
        }

        if (!isEnergyMaterial() && _parameters.material != CreationMaterial_Fluid) {
            DescEditService::get().reconnectObjects(newEntities, 1.5f);
        }
        _SimulationFacade::get()->addAndSelectSimulationData(std::move(newEntities));

        if (!isEnergyMaterial() && _parameters.material != CreationMaterial_Fluid) {
            _SimulationFacade::get()->reconnectSelectedObjects();
        }
    }
    EditorModel::get().update();
}

void CreatorController::finishDrawing()
{
    _drawingDescription.clear();
    _drawingOccupancy.clear();
}

void CreatorController::processPlacementPreview()
{
    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }

    PreviewKey key{
        .tool = EditorModel::get().getTool(),
        .material = _parameters.material,
        .rectHorizontalObjects = _parameters.rectHorizontalObjects,
        .rectVerticalObjects = _parameters.rectVerticalObjects,
        .layers = _parameters.layers,
        .outerRadius = _parameters.outerRadius,
        .innerRadius = _parameters.innerRadius,
        .objectDistance = _parameters.objectDistance,
    };
    if (_previewKey != key) {
        _previewKey = key;
        auto shape = createShape({0, 0});
        _previewPositions.clear();
        for (auto const& object : shape._objects) {
            _previewPositions.emplace_back(object._pos);
        }
        for (auto const& energy : shape._energies) {
            _previewPositions.emplace_back(energy._pos);
        }
    }
    if (_previewPositions.empty()) {
        return;
    }

    auto drawList = ImGui::GetBackgroundDrawList();
    auto mousePos = ImGui::GetMousePos();
    auto zoom = Viewport::get().getZoomFactor();
    if (toInt(_previewPositions.size()) > MaxNumPlacementPreviewPoints) {
        auto topLeft = _previewPositions.front();
        auto bottomRight = _previewPositions.front();
        for (auto const& pos : _previewPositions) {
            topLeft = {std::min(topLeft.x, pos.x), std::min(topLeft.y, pos.y)};
            bottomRight = {std::max(bottomRight.x, pos.x), std::max(bottomRight.y, pos.y)};
        }
        drawList->AddRect(
            {mousePos.x + topLeft.x * zoom, mousePos.y + topLeft.y * zoom},
            {mousePos.x + bottomRight.x * zoom, mousePos.y + bottomRight.y * zoom},
            Const::ConstructionPreviewLineColor,
            0,
            0,
            scale(PreviewLineThickness));
        return;
    }
    auto radius = std::max(scale(PlacementPreviewMinRadius), zoom * PlacementPreviewRadiusFactor);
    for (auto const& pos : _previewPositions) {
        drawList->AddCircleFilled({mousePos.x + pos.x * zoom, mousePos.y + pos.y * zoom}, radius, Const::ConstructionPreviewPointColor);
    }
}

void CreatorController::processPointPreview(std::vector<RealVector2D> const& path, bool closed) const
{
    if (_points.empty()) {
        return;
    }

    auto drawList = ImGui::GetBackgroundDrawList();
    auto viewPath = mapToViewPositions(path);
    drawList->AddPolyline(
        viewPath.data(),
        toInt(viewPath.size()),
        Const::ConstructionPreviewLineColor,
        closed ? ImDrawFlags_Closed : ImDrawFlags_None,
        scale(PreviewLineThickness));

    if (!ImGui::GetIO().WantCaptureMouse) {
        auto mousePos = ImGui::GetMousePos();
        drawList->AddLine(mapToViewPosition(_points.back()), mousePos, Const::ConstructionPreviewHintLineColor, scale(PreviewLineThickness));
        if (closed && _points.size() > 1) {
            drawList->AddLine(mousePos, mapToViewPosition(_points.front()), Const::ConstructionPreviewHintLineColor, scale(PreviewLineThickness));
        }
    }

    for (auto const& point : _points) {
        drawList->AddCircleFilled(mapToViewPosition(point), scale(PreviewPointRadius), Const::ConstructionPreviewPointColor);
    }
}

void CreatorController::processControlPolygonPreview() const
{
    if (_points.size() < 2) {
        return;
    }

    auto viewPath = mapToViewPositions(_points);
    ImGui::GetBackgroundDrawList()->AddPolyline(
        viewPath.data(), toInt(viewPath.size()), Const::ConstructionPreviewHintLineColor, ImDrawFlags_None, scale(PreviewLineThickness));
}

ContentDesc CreatorController::createShape(RealVector2D const& center) const
{
    switch (EditorModel::get().getTool()) {
    case EditTool_Object:
        return CreatorService::get().createSingleObject(getObjectProperties(), center);
    case EditTool_Rectangle:
        return CreatorService::get().createRectangle(
            getObjectProperties(), center, {_parameters.rectHorizontalObjects, _parameters.rectVerticalObjects}, _parameters.objectDistance);
    case EditTool_Hexagon:
        return CreatorService::get().createHexagon(getObjectProperties(), center, _parameters.layers, _parameters.objectDistance);
    case EditTool_Disc:
        return CreatorService::get().createDisc(getObjectProperties(), center, _parameters.outerRadius, _parameters.innerRadius, _parameters.objectDistance);
    default:
        return ContentDesc();
    }
}

void CreatorController::addToSimulation(ContentDesc&& content) const
{
    if (!content.isEmpty()) {
        _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
    }
}

void CreatorController::validateAndCorrect()
{
    _parameters.energy = std::max(0.0f, _parameters.energy);
    _parameters.stiffness = std::min(1.0f, std::max(0.0f, _parameters.stiffness));
    _parameters.material =
        std::max(static_cast<int>(CreationMaterial_Solid), std::min(static_cast<int>(CreationMaterial_EnergyParticle), _parameters.material));
    _parameters.objectDistance = std::min(10.0f, std::max(0.5f, _parameters.objectDistance));
    _parameters.rectHorizontalObjects = std::max(1, _parameters.rectHorizontalObjects);
    _parameters.rectVerticalObjects = std::max(1, _parameters.rectVerticalObjects);
    _parameters.layers = std::max(1, _parameters.layers);
    _parameters.innerRadius = std::max(0.0f, _parameters.innerRadius);
    _parameters.outerRadius = std::max(_parameters.innerRadius, _parameters.outerRadius);
}

bool CreatorController::isEnergyMaterial() const
{
    return _parameters.material == CreationMaterial_EnergyParticle;
}

int CreatorController::getMinNumPoints() const
{
    return EditorModel::get().getTool() == EditTool_Polygon ? 3 : 2;
}

CreatorService::ObjectProperties CreatorController::getObjectProperties() const
{
    return CreatorService::ObjectProperties()
        .material(_parameters.material)
        .color(EditorModel::get().getDefaultColorCode())
        .energy(_parameters.energy)
        .stiffness(_parameters.stiffness)
        .glow(_parameters.glow)
        .isStatic(_parameters.isStatic)
        .sticky(_parameters.sticky);
}
