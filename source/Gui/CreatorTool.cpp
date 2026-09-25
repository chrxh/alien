#include "CreatorTool.h"

#include <algorithm>

#include <imgui.h>

#include <Base/GlobalSettings.h>
#include <Base/Math.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "EntityAttributeHelp.h"
#include "HelpStrings.h"
#include "SimulationInteractionController.h"
#include "StyleService.h"
#include "Viewport.h"

namespace
{
    auto constexpr RightColumnWidth = 140.0f;
    auto constexpr PointButtonWidth = 70.0f;
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
}

void CreatorTool::init()
{
    auto& settings = GlobalSettings::get();
    _energy = settings.getValue("editors.creator.energy", _energy);
    _stiffness = settings.getValue("editors.creator.stiffness", _stiffness);
    _static = settings.getValue("editors.creator.static", _static);
    _objectDistance = settings.getValue("editors.creator.object distance", _objectDistance);
    _glow = settings.getValue("editors.creator.glow", _glow);
    _makeSticky = settings.getValue("editors.creator.make sticky", _makeSticky);
    _rectHorizontalObjects = settings.getValue("editors.creator.rect horizontal objects", _rectHorizontalObjects);
    _rectVerticalObjects = settings.getValue("editors.creator.rect vertical objects", _rectVerticalObjects);
    _layers = settings.getValue("editors.creator.layers", _layers);
    _outerRadius = settings.getValue("editors.creator.outer radius", _outerRadius);
    _innerRadius = settings.getValue("editors.creator.inner radius", _innerRadius);
    _material = settings.getValue("editors.creator.material", _material);
}

void CreatorTool::shutdown()
{
    auto& settings = GlobalSettings::get();
    settings.setValue("editors.creator.energy", _energy);
    settings.setValue("editors.creator.stiffness", _stiffness);
    settings.setValue("editors.creator.static", _static);
    settings.setValue("editors.creator.object distance", _objectDistance);
    settings.setValue("editors.creator.glow", _glow);
    settings.setValue("editors.creator.make sticky", _makeSticky);
    settings.setValue("editors.creator.rect horizontal objects", _rectHorizontalObjects);
    settings.setValue("editors.creator.rect vertical objects", _rectVerticalObjects);
    settings.setValue("editors.creator.layers", _layers);
    settings.setValue("editors.creator.outer radius", _outerRadius);
    settings.setValue("editors.creator.inner radius", _innerRadius);
    settings.setValue("editors.creator.material", _material);
}

void CreatorTool::process()
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
        processPointPreview(CreatorService::get().calcBezierCurvePath(_points, _objectDistance), false);
        break;
    case EditTool_Polygon:
        processPointPreview(_points, true);
        break;
    default:
        break;
    }
}

void CreatorTool::processOptions()
{
    auto tool = EditorModel::get().getTool();
    processColorWidget();
    if (tool == EditTool_Freehand) {
        auto pencilWidth = EditorModel::get().getPencilWidth();
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters()
                .name("Pencil radius")
                .min(1.0f)
                .max(8.0f)
                .textWidth(RightColumnWidth)
                .format("%.1f")
                .tooltip(Const::CreatorPencilRadiusTooltip),
            &pencilWidth);
        EditorModel::get().setPencilWidth(pencilWidth);
    }
    processMaterialWidgets();
    processShapeWidgets();
    if (tool != EditTool_Object && tool != EditTool_Freehand) {
        processObjectDistanceWidget();
    }
    if (tool != EditTool_Object) {
        processStickyWidget();
    }
    processStaticWidget();
    if (tool == EditTool_Line || tool == EditTool_Curve || tool == EditTool_Polygon) {
        processPointButtons();
    }
    validateAndCorrect();
}

void CreatorTool::onPlace(RealVector2D const& worldPos)
{
    addToSimulation(createShape(worldPos));
    EditorModel::get().update();
}

void CreatorTool::onAddPoint(RealVector2D const& worldPos)
{
    _points.emplace_back(worldPos);
}

void CreatorTool::onRemoveLastPoint()
{
    if (!_points.empty()) {
        _points.pop_back();
    }
}

bool CreatorTool::isFinishingPointsPossible() const
{
    return toInt(_points.size()) >= getMinNumPoints();
}

void CreatorTool::onFinishPoints()
{
    if (!isFinishingPointsPossible()) {
        return;
    }
    auto tool = EditorModel::get().getTool();
    if (tool == EditTool_Line) {
        addToSimulation(CreatorService::get().createLine(getObjectProperties(), _points, _objectDistance));
    } else if (tool == EditTool_Curve) {
        addToSimulation(CreatorService::get().createCurve(getObjectProperties(), _points, _objectDistance));
    } else if (tool == EditTool_Polygon) {
        addToSimulation(CreatorService::get().createPolygon(getObjectProperties(), _points, _objectDistance));
    }
    _points.clear();
    EditorModel::get().update();
}

bool CreatorTool::hasPoints() const
{
    return !_points.empty();
}

void CreatorTool::onAbortPoints()
{
    _points.clear();
}

void CreatorTool::onDrawing()
{
    auto mousePos = ImGui::GetMousePos();
    auto pos = Viewport::get().mapViewToWorldPosition({mousePos.x, mousePos.y});

    auto createAlignedCircle = [&](RealVector2D const& pos) {
        return CreatorService::get().createPencilDot(getObjectProperties(), pos, EditorModel::get().getPencilWidth());
    };

    auto prevEntityCount = isEnergyMaterial() ? _drawingDescription._energies.size() : _drawingDescription._objects.size();

    if (_drawingDescription.isEmpty()) {
        DescEditService::get().addIfSpaceAvailable(
            _drawingDescription, _drawingOccupancy, createAlignedCircle(pos), 0.5f, _SimulationFacade::get()->getWorldSize());
        _lastDrawPos = pos;
    } else {
        auto posDelta = Math::length(pos - _lastDrawPos);
        if (posDelta > 0) {
            auto lastDrawPos = _lastDrawPos;
            for (float interDelta = 0; interDelta < posDelta; interDelta += 1.0f) {
                auto drawPos = lastDrawPos + (pos - lastDrawPos) * interDelta / posDelta;
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

        if (!isEnergyMaterial() && _material != CreationMaterial_Fluid) {
            DescEditService::get().reconnectObjects(newEntities, 1.5f);
        }
        _SimulationFacade::get()->addAndSelectSimulationData(std::move(newEntities));

        if (!isEnergyMaterial() && _material != CreationMaterial_Fluid) {
            _SimulationFacade::get()->reconnectSelectedObjects();
        }
    }
    EditorModel::get().update();
}

void CreatorTool::finishDrawing()
{
    _drawingDescription.clear();
    _drawingOccupancy.clear();
}

void CreatorTool::processShapeWidgets()
{
    auto tool = EditorModel::get().getTool();
    if (tool == EditTool_Rectangle) {
        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Horizontal objects").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleWidthTooltip),
            _rectHorizontalObjects);
        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Vertical objects").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleHeightTooltip),
            _rectVerticalObjects);
    }
    if (tool == EditTool_Hexagon) {
        AlienGui::InputInt(AlienGui::InputIntParameters().name("Layers").textWidth(RightColumnWidth).tooltip(Const::CreatorHexagonLayersTooltip), _layers);
    }
    if (tool == EditTool_Disc) {
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Outer radius").textWidth(RightColumnWidth).format("%.0f").tooltip(Const::CreatorDiscOuterRadiusTooltip),
            _outerRadius);
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Inner radius").textWidth(RightColumnWidth).format("%.0f").tooltip(Const::CreatorDiscInnerRadiusTooltip),
            _innerRadius);
    }
}

void CreatorTool::processColorWidget()
{
    auto color = EditorModel::get().getDefaultColorCode();
    AlienGui::ComboColor(
        AlienGui::ComboColorParameters()
            .customizationColors(_SimulationFacade::get()->getSimulationParameters().customizationColors.value)
            .name("Color")
            .textWidth(RightColumnWidth)
            .tooltip(EntityAttributeHelp::get(EntityAttribute::Color)),
        color);
    EditorModel::get().setDefaultColorCode(color);
}

void CreatorTool::processMaterialWidgets()
{
    AlienGui::Switcher(
        AlienGui::SwitcherParameters()
            .name("Material")
            .textWidth(RightColumnWidth)
            .values({"Solid", "Fluid", "Free cells", "Energy particles"})
            .tooltip(Const::CreatorDrawingTypeTooltip),
        &_material);
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(RightColumnWidth).tooltip(Const::CellEnergyTooltip), _energy);
    if (_material == CreationMaterial_Fluid) {
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters()
                .name("Glow")
                .min(0)
                .max(1.0f)
                .format("%.2f")
                .textWidth(RightColumnWidth)
                .tooltip(EntityAttributeHelp::get(EntityAttribute::FluidGlow)),
            &_glow);
    }
    if (!isEnergyMaterial() && _material != CreationMaterial_Fluid) {
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters()
                .name("Stiffness")
                .max(1.0f)
                .min(0.0f)
                .textWidth(RightColumnWidth)
                .tooltip(EntityAttributeHelp::get(EntityAttribute::Stiffness)),
            &_stiffness);
    }
}

void CreatorTool::processObjectDistanceWidget()
{
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Object distance").format("%.2f").step(0.1).textWidth(RightColumnWidth).tooltip(Const::CreatorDistanceTooltip),
        _objectDistance);
}

void CreatorTool::processStickyWidget()
{
    AlienGui::Checkbox(
        AlienGui::CheckboxParameters().name("Sticky").textWidth(RightColumnWidth).tooltip(EntityAttributeHelp::get(EntityAttribute::Sticky)), &_makeSticky);
}

void CreatorTool::processStaticWidget()
{
    if (!isEnergyMaterial()) {
        AlienGui::Checkbox(
            AlienGui::CheckboxParameters().name("Static").textWidth(RightColumnWidth).tooltip(EntityAttributeHelp::get(EntityAttribute::Static)), &_static);
    }
}

void CreatorTool::processPointButtons()
{
    AlienGui::Separator();

    ImGui::BeginDisabled(!isFinishingPointsPossible());
    if (AlienGui::Button("Finish", PointButtonWidth)) {
        onFinishPoints();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!hasPoints());
    if (AlienGui::Button("Abort", PointButtonWidth)) {
        onAbortPoints();
    }
    ImGui::EndDisabled();
}

void CreatorTool::processPlacementPreview()
{
    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }

    PreviewKey key{
        .tool = EditorModel::get().getTool(),
        .material = _material,
        .rectHorizontalObjects = _rectHorizontalObjects,
        .rectVerticalObjects = _rectVerticalObjects,
        .layers = _layers,
        .outerRadius = _outerRadius,
        .innerRadius = _innerRadius,
        .objectDistance = _objectDistance,
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

void CreatorTool::processPointPreview(std::vector<RealVector2D> const& path, bool closed) const
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

void CreatorTool::processControlPolygonPreview() const
{
    if (_points.size() < 2) {
        return;
    }

    auto viewPath = mapToViewPositions(_points);
    ImGui::GetBackgroundDrawList()->AddPolyline(
        viewPath.data(), toInt(viewPath.size()), Const::ConstructionPreviewHintLineColor, ImDrawFlags_None, scale(PreviewLineThickness));
}

ContentDesc CreatorTool::createShape(RealVector2D const& center) const
{
    switch (EditorModel::get().getTool()) {
    case EditTool_Object:
        return CreatorService::get().createSingleObject(getObjectProperties(), center);
    case EditTool_Rectangle:
        return CreatorService::get().createRectangle(getObjectProperties(), center, {_rectHorizontalObjects, _rectVerticalObjects}, _objectDistance);
    case EditTool_Hexagon:
        return CreatorService::get().createHexagon(getObjectProperties(), center, _layers, _objectDistance);
    case EditTool_Disc:
        return CreatorService::get().createDisc(getObjectProperties(), center, _outerRadius, _innerRadius, _objectDistance);
    default:
        return ContentDesc();
    }
}

void CreatorTool::addToSimulation(ContentDesc&& content) const
{
    if (!content.isEmpty()) {
        _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
    }
}

void CreatorTool::validateAndCorrect()
{
    _energy = std::max(0.0f, _energy);
    _stiffness = std::min(1.0f, std::max(0.0f, _stiffness));
    _material = std::max(static_cast<int>(CreationMaterial_Solid), std::min(static_cast<int>(CreationMaterial_EnergyParticle), _material));
    _objectDistance = std::min(10.0f, std::max(0.5f, _objectDistance));
    _rectHorizontalObjects = std::max(1, _rectHorizontalObjects);
    _rectVerticalObjects = std::max(1, _rectVerticalObjects);
    _layers = std::max(1, _layers);
    _innerRadius = std::max(0.0f, _innerRadius);
    _outerRadius = std::max(_innerRadius, _outerRadius);
}

bool CreatorTool::isEnergyMaterial() const
{
    return _material == CreationMaterial_EnergyParticle;
}

int CreatorTool::getMinNumPoints() const
{
    return EditorModel::get().getTool() == EditTool_Polygon ? 3 : 2;
}

CreatorService::ObjectProperties CreatorTool::getObjectProperties() const
{
    return CreatorService::ObjectProperties()
        .material(_material)
        .color(EditorModel::get().getDefaultColorCode())
        .energy(_energy)
        .stiffness(_stiffness)
        .glow(_glow)
        .isStatic(_static)
        .sticky(_makeSticky);
}
