#include "CreatorWindow.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ranges>
#include <span>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/Math.h>

#include <EngineInterface/DescEditService.h>
#include <EngineInterface/Descs.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "EditorController.h"
#include "EditorModel.h"
#include "HelpStrings.h"
#include "ImageToPatternDialog.h"
#include "SimulationInteractionController.h"
#include "StyleRepository.h"
#include "Viewport.h"

#include "Fonts/AlienIconFont.h"

namespace
{
    auto const ModeText = std::unordered_map<CreationMode, std::string>{
        {CreationMode_CreateObject, "Create a single object"},
        {CreationMode_CreateRectangle, "Create a rectangular object network"},
        {CreationMode_CreateHexagon, "Create a hexagonal object network"},
        {CreationMode_CreateDisc, "Create a disc-shaped object network"},
        {CreationMode_CreateLine, "Create an object network along a line"},
        {CreationMode_CreateCurve, "Create an object network along a Bezier curve"},
        {CreationMode_CreatePolygon, "Create a polygon-shaped object network"},
        {CreationMode_Drawing, "Draw freehand"},
    };

    auto const ToolbarModeIcons = std::vector<std::pair<CreationMode, std::string>>{
        {CreationMode_CreateObject, ICON_DOT},
        {CreationMode_CreateRectangle, ICON_RECTANGLE},
        {CreationMode_CreateHexagon, ICON_HEXAGON},
        {CreationMode_CreateDisc, ICON_DISC},
        {CreationMode_CreateLine, ICON_FA_SLASH},
        {CreationMode_CreateCurve, ICON_FA_BEZIER_CURVE},
        {CreationMode_CreatePolygon, ICON_FA_DRAW_POLYGON},
        {CreationMode_Drawing, ICON_FA_PAINT_BRUSH},
    };

    auto const RightColumnWidth = 160.0f;
    auto const PointButtonWidth = 60.0f;
    auto constexpr MaxNumObjects = size_t{1000000};
    auto constexpr PreviewLineThickness = 2.0f;
    auto constexpr PreviewPointRadius = 3.0f;

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

    RealVector2D evaluateBezier(std::vector<RealVector2D> const& controlPoints, float t)
    {
        auto points = controlPoints;
        for (auto count = points.size(); count > 1; --count) {
            auto range = std::span(points).first(count);
            for (auto&& [current, next] : std::views::zip(range, range | std::views::drop(1))) {
                current = current * (1.0f - t) + next * t;
            }
        }
        return points.front();
    }

    std::vector<RealVector2D> distributeAlongPath(std::vector<RealVector2D> const& path, float distance)
    {
        std::vector<RealVector2D> result;
        if (path.size() < 2 || distance < NEAR_ZERO) {
            return result;
        }
        result.emplace_back(path.front());
        auto pendingDistance = distance;
        for (auto const& [from, to] : std::views::zip(path, path | std::views::drop(1))) {
            auto segmentLength = Math::length(to - from);
            if (segmentLength < NEAR_ZERO) {
                continue;
            }
            auto direction = (to - from) / segmentLength;
            auto offset = pendingDistance;
            for (; offset < segmentLength + NEAR_ZERO && result.size() < MaxNumObjects; offset += distance) {
                result.emplace_back(from + direction * offset);
            }
            pendingDistance = offset - segmentLength;
        }
        return result;
    }

    bool isInsidePolygon(std::vector<RealVector2D> const& closedPolygon, RealVector2D const& pos)
    {
        auto result = false;
        for (auto const& [from, to] : std::views::zip(closedPolygon, closedPolygon | std::views::drop(1))) {
            if ((from.y > pos.y) != (to.y > pos.y)) {
                auto intersectionX = from.x + (pos.y - from.y) / (to.y - from.y) * (to.x - from.x);
                if (pos.x < intersectionX) {
                    result = !result;
                }
            }
        }
        return result;
    }

    std::vector<RealVector2D> distributeHexagonallyInPolygon(std::vector<RealVector2D> const& polygon, float distance)
    {
        std::vector<RealVector2D> result;
        if (polygon.size() < 3 || distance < NEAR_ZERO) {
            return result;
        }
        auto closedPolygon = polygon;
        closedPolygon.emplace_back(polygon.front());

        auto minPos = polygon.front();
        auto maxPos = polygon.front();
        for (auto const& point : polygon) {
            minPos.x = std::min(minPos.x, point.x);
            minPos.y = std::min(minPos.y, point.y);
            maxPos.x = std::max(maxPos.x, point.x);
            maxPos.y = std::max(maxPos.y, point.y);
        }

        auto rowDistance = distance * sqrtf(3.0f) / 2;
        auto rowIndex = 0;
        for (auto y = minPos.y; y < maxPos.y + NEAR_ZERO && result.size() < MaxNumObjects; y += rowDistance) {
            auto rowOffset = rowIndex % 2 == 0 ? 0.0f : distance / 2;
            for (auto x = minPos.x + rowOffset; x < maxPos.x + NEAR_ZERO && result.size() < MaxNumObjects; x += distance) {
                if (isInsidePolygon(closedPolygon, {x, y})) {
                    result.emplace_back(RealVector2D{x, y});
                }
            }
            ++rowIndex;
        }
        return result;
    }
}

void CreatorWindow::initIntern()
{
    _energy = GlobalSettings::get().getValue("editors.creator.energy", _energy);
    _stiffness = GlobalSettings::get().getValue("editors.creator.stiffness", _stiffness);
    _static = GlobalSettings::get().getValue("editors.creator.static", _static);
    _objectDistance = GlobalSettings::get().getValue("editors.creator.object distance", _objectDistance);
    _glow = GlobalSettings::get().getValue("editors.creator.glow", _glow);
    _makeSticky = GlobalSettings::get().getValue("editors.creator.make sticky", _makeSticky);
    _rectHorizontalObjects = GlobalSettings::get().getValue("editors.creator.rect horizontal objects", _rectHorizontalObjects);
    _rectVerticalObjects = GlobalSettings::get().getValue("editors.creator.rect vertical objects", _rectVerticalObjects);
    _layers = GlobalSettings::get().getValue("editors.creator.layers", _layers);
    _outerRadius = GlobalSettings::get().getValue("editors.creator.outer radius", _outerRadius);
    _innerRadius = GlobalSettings::get().getValue("editors.creator.inner radius", _innerRadius);
    _material = GlobalSettings::get().getValue("editors.creator.material", _material);
    _mode = GlobalSettings::get().getValue("editors.creator.mode", _mode);
}

void CreatorWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("editors.creator.energy", _energy);
    GlobalSettings::get().setValue("editors.creator.stiffness", _stiffness);
    GlobalSettings::get().setValue("editors.creator.static", _static);
    GlobalSettings::get().setValue("editors.creator.object distance", _objectDistance);
    GlobalSettings::get().setValue("editors.creator.glow", _glow);
    GlobalSettings::get().setValue("editors.creator.make sticky", _makeSticky);
    GlobalSettings::get().setValue("editors.creator.rect horizontal objects", _rectHorizontalObjects);
    GlobalSettings::get().setValue("editors.creator.rect vertical objects", _rectVerticalObjects);
    GlobalSettings::get().setValue("editors.creator.layers", _layers);
    GlobalSettings::get().setValue("editors.creator.outer radius", _outerRadius);
    GlobalSettings::get().setValue("editors.creator.inner radius", _innerRadius);
    GlobalSettings::get().setValue("editors.creator.material", _material);
    GlobalSettings::get().setValue("editors.creator.mode", _mode);
}


void CreatorWindow::processIntern()
{
    processToolbar();
    updateInteractionMode();

    switch (_mode) {
    case CreationMode_CreateObject:
        processCreateObject();
        break;
    case CreationMode_CreateRectangle:
        processCreateRectangle();
        break;
    case CreationMode_CreateHexagon:
        processCreateHexagon();
        break;
    case CreationMode_CreateDisc:
        processCreateDisc();
        break;
    case CreationMode_Drawing:
        processDrawing();
        break;
    case CreationMode_CreateLine:
        processCreateLine();
        break;
    case CreationMode_CreateCurve:
        processCreateCurve();
        break;
    case CreationMode_CreatePolygon:
        processCreatePolygon();
        break;
    }

    validateAndCorrect();
}

void CreatorWindow::processCreateObject()
{
    if (beginParameterPanel()) {
        processColorWidget();
        processMaterialWidgets();
        processStaticWidget();
    }
    endParameterPanel();

    if (processBuildButton()) {
        createSingleObject();
        EditorModel::get().update();
    }
}

void CreatorWindow::processCreateRectangle()
{
    if (beginParameterPanel()) {
        processColorWidget();
        processMaterialWidgets();
        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Horizontal objects").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleWidthTooltip),
            _rectHorizontalObjects);
        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Vertical objects").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleHeightTooltip),
            _rectVerticalObjects);
        processObjectDistanceWidget();
        processStickyWidget();
        processStaticWidget();
    }
    endParameterPanel();

    if (processBuildButton()) {
        createRectangle();
        EditorModel::get().update();
    }
}

void CreatorWindow::processCreateHexagon()
{
    if (beginParameterPanel()) {
        processColorWidget();
        processMaterialWidgets();
        AlienGui::InputInt(AlienGui::InputIntParameters().name("Layers").textWidth(RightColumnWidth).tooltip(Const::CreatorHexagonLayersTooltip), _layers);
        processObjectDistanceWidget();
        processStickyWidget();
        processStaticWidget();
    }
    endParameterPanel();

    if (processBuildButton()) {
        createHexagon();
        EditorModel::get().update();
    }
}

void CreatorWindow::processCreateDisc()
{
    if (beginParameterPanel()) {
        processColorWidget();
        processMaterialWidgets();
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Outer radius").textWidth(RightColumnWidth).format("%.0f").tooltip(Const::CreatorDiscOuterRadiusTooltip),
            _outerRadius);
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Inner radius").textWidth(RightColumnWidth).format("%.0f").tooltip(Const::CreatorDiscInnerRadiusTooltip),
            _innerRadius);
        processObjectDistanceWidget();
        processStickyWidget();
        processStaticWidget();
    }
    endParameterPanel();

    if (processBuildButton()) {
        createDisc();
        EditorModel::get().update();
    }
}

void CreatorWindow::processDrawing()
{
    if (beginParameterPanel()) {
        processColorWidget();

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

        processMaterialWidgets();
        processStickyWidget();
        processStaticWidget();
    }
    endParameterPanel();
}

void CreatorWindow::processCreateLine()
{
    if (beginParameterPanel()) {
        processColorWidget();
        processMaterialWidgets();
        processObjectDistanceWidget();
        processStickyWidget();
        processStaticWidget();
    }
    endParameterPanel();

    processPointPreview(_points, false);

    if (processPointButtons(2)) {
        createObjectNetwork(distributeAlongPath(_points, _objectDistance), _objectDistance * 1.5f);
        _points.clear();
        EditorModel::get().update();
    }
}

void CreatorWindow::processCreateCurve()
{
    if (beginParameterPanel()) {
        processColorWidget();
        processMaterialWidgets();
        processObjectDistanceWidget();
        processStickyWidget();
        processStaticWidget();
    }
    endParameterPanel();

    auto path = calcBezierCurvePath();
    processControlPolygonPreview();
    processPointPreview(path, false);

    if (processPointButtons(2)) {
        createObjectNetwork(distributeAlongPath(path, _objectDistance), _objectDistance * 1.5f);
        _points.clear();
        EditorModel::get().update();
    }
}

void CreatorWindow::processCreatePolygon()
{
    if (beginParameterPanel()) {
        processColorWidget();
        processMaterialWidgets();
        processObjectDistanceWidget();
        processStickyWidget();
        processStaticWidget();
    }
    endParameterPanel();

    processPointPreview(_points, true);

    if (processPointButtons(3)) {
        createObjectNetwork(distributeHexagonallyInPolygon(_points, _objectDistance), _objectDistance * 1.7f);
        _points.clear();
        EditorModel::get().update();
    }
}

void CreatorWindow::updateInteractionMode()
{
    auto& simInteractionController = SimulationInteractionController::get();
    if (simInteractionController.getInteractionMode() == InteractionMode_PositionSelection) {
        abortPointPlacement();
    } else {
        simInteractionController.setInteractionMode(getInteractionMode());
    }
}

bool CreatorWindow::beginParameterPanel()
{
    auto result = ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50.0f)), false, ImGuiWindowFlags_HorizontalScrollbar);
    if (result) {
        AlienGui::Group(AlienGui::GroupParameters().text(ModeText.at(_mode)));
    }
    return result;
}

void CreatorWindow::endParameterPanel()
{
    ImGui::EndChild();
}

void CreatorWindow::processColorWidget()
{
    auto color = EditorModel::get().getDefaultColorCode();
    AlienGui::ComboColor(
        AlienGui::ComboColorParameters()
            .customizationColors(_SimulationFacade::get()->getSimulationParameters().customizationColors.value)
            .name("Color")
            .textWidth(RightColumnWidth)
            .tooltip(Const::GenomeColorTooltip),
        color);
    EditorModel::get().setDefaultColorCode(color);
}

void CreatorWindow::processMaterialWidgets()
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
        AlienGui::SliderFloat(AlienGui::SliderFloatParameters().name("Glow").min(0).max(1.0f).format("%.2f").textWidth(RightColumnWidth), &_glow);
    }
    if (!isEnergyMaterial() && _material != CreationMaterial_Fluid) {
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Stiffness").max(1.0f).min(0.0f).textWidth(RightColumnWidth).tooltip(Const::CellStiffnessTooltip),
            &_stiffness);
    }
}

void CreatorWindow::processObjectDistanceWidget()
{
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Object distance").format("%.2f").step(0.1).textWidth(RightColumnWidth).tooltip(Const::CreatorDistanceTooltip),
        _objectDistance);
}

void CreatorWindow::processStickyWidget()
{
    AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Sticky").textWidth(RightColumnWidth).tooltip(Const::CreatorStickyTooltip), _makeSticky);
}

void CreatorWindow::processStaticWidget()
{
    if (!isEnergyMaterial()) {
        AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Static").textWidth(RightColumnWidth).tooltip(Const::CellStaticTooltip), _static);
    }
}

bool CreatorWindow::processBuildButton()
{
    AlienGui::Separator();
    return AlienGui::Button("Build");
}

bool CreatorWindow::processPointButtons(int minNumPoints)
{
    AlienGui::Separator();

    auto result = false;
    if (!_pointPlacementActive) {
        if (AlienGui::Button("Start", PointButtonWidth)) {
            _pointPlacementActive = true;
        }
    } else {
        ImGui::BeginDisabled(toInt(_points.size()) < minNumPoints);
        if (AlienGui::Button("Finish", PointButtonWidth)) {
            _pointPlacementActive = false;
            result = true;
        }
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    ImGui::BeginDisabled(!_pointPlacementActive);
    if (AlienGui::Button("Abort", PointButtonWidth)) {
        abortPointPlacement();
    }
    ImGui::EndDisabled();

    return result;
}

void CreatorWindow::abortPointPlacement()
{
    _pointPlacementActive = false;
    _points.clear();
}

void CreatorWindow::processToolbar()
{
    auto previousMode = _mode;

    for (auto const& [mode, icon] : ToolbarModeIcons) {
        AlienGui::SelectableToolbarButton(icon, _mode, mode, mode);
        AlienGui::Tooltip(ModeText.at(mode));
        ImGui::SameLine();
    }

    AlienGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_IMAGE).tooltip("Create a pattern from an image"))) {
        ImageToPatternDialog::get().show();
    }

    if (_mode != previousMode) {
        abortPointPlacement();
    }
}

void CreatorWindow::processPointPreview(std::vector<RealVector2D> const& path, bool closed) const
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

void CreatorWindow::processControlPolygonPreview() const
{
    if (_points.size() < 2) {
        return;
    }

    auto viewPath = mapToViewPositions(_points);
    ImGui::GetBackgroundDrawList()->AddPolyline(
        viewPath.data(), toInt(viewPath.size()), Const::ConstructionPreviewHintLineColor, ImDrawFlags_None, scale(PreviewLineThickness));
}

void CreatorWindow::processBackground()
{
    if (!isShown()) {
        abortPointPlacement();
    }
}

bool CreatorWindow::isShown()
{
    return _on && EditorController::get().isOn();
}

void CreatorWindow::onDrawing()
{
    auto mousePos = ImGui::GetMousePos();
    auto pos = Viewport::get().mapViewToWorldPosition({mousePos.x, mousePos.y});

    auto createAlignedCircle = [&](auto pos) {
        if (EditorModel::get().getPencilWidth() > 1 + NEAR_ZERO) {
            pos.x = toFloat(toInt(pos.x));
            pos.y = toFloat(toInt(pos.y));
        }
        auto desc = DescEditService::get().createCircle(DescEditService::CreateCircleParameters()
                                                            .center(pos)
                                                            .radius(EditorModel::get().getPencilWidth())
                                                            .type(isEnergyMaterial() ? ObjectTypeDesc{SolidDesc()} : getObjectTypeDesc())
                                                            .stiffness(_stiffness)
                                                            .sticky(_makeSticky)
                                                            .cellDistance(1.0f)
                                                            .color(EditorModel::get().getDefaultColorCode())
                                                            .isStatic(_static)
                                                            .connectObjects(false));
        return isEnergyMaterial() ? convertToEnergyParticles(desc) : desc;
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

void CreatorWindow::finishDrawing()
{
    _drawingDescription.clear();
    _drawingOccupancy.clear();
}

void CreatorWindow::onAddPoint(RealVector2D const& worldPos)
{
    _points.emplace_back(worldPos);
}

void CreatorWindow::onRemoveLastPoint()
{
    if (!_points.empty()) {
        _points.pop_back();
    }
}

CreatorWindow::CreatorWindow()
    : AlienWindow("Creator", "editors.creator", false, false, {464.0f, 61.0f}, {400.0f, 370.0f})
{}

void CreatorWindow::createSingleObject()
{
    ContentDesc description;
    if (isEnergyMaterial()) {
        description._energies.emplace_back(EnergyDesc().pos(getRandomPos()).energy(_energy).color(EditorModel::get().getDefaultColorCode()));
    } else {
        description._objects.emplace_back(ObjectDesc()
                                              .pos(getRandomPos())
                                              .stiffness(_stiffness)
                                              .color(EditorModel::get().getDefaultColorCode())
                                              .isStatic(_static)
                                              .sticky(_makeSticky)
                                              .type(getObjectTypeDesc()));
    }
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

void CreatorWindow::createRectangle()
{
    if (_rectHorizontalObjects <= 0 || _rectVerticalObjects <= 0) {
        return;
    }

    auto description = DescEditService::get().createRect(DescEditService::CreateRectParameters()
                                                             .objectType(isEnergyMaterial() ? ObjectTypeDesc{SolidDesc()} : getObjectTypeDesc())
                                                             .width(_rectHorizontalObjects)
                                                             .height(_rectVerticalObjects)
                                                             .cellDistance(_objectDistance)
                                                             .stiffness(_stiffness)
                                                             .sticky(_makeSticky)
                                                             .color(EditorModel::get().getDefaultColorCode())
                                                             .center(getRandomPos())
                                                             .isStatic(_static));
    if (isEnergyMaterial()) {
        description = convertToEnergyParticles(description);
    }
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

void CreatorWindow::createHexagon()
{
    if (_layers <= 0) {
        return;
    }

    auto description = DescEditService::get().createHex(DescEditService::CreateHexParameters()
                                                            .objectType(isEnergyMaterial() ? ObjectTypeDesc{SolidDesc()} : getObjectTypeDesc())
                                                            .layers(_layers)
                                                            .cellDistance(_objectDistance)
                                                            .stiffness(_stiffness)
                                                            .sticky(_makeSticky)
                                                            .color(EditorModel::get().getDefaultColorCode())
                                                            .center(getRandomPos())
                                                            .isStatic(_static));
    if (isEnergyMaterial()) {
        description = convertToEnergyParticles(description);
    } else {
        DescEditService::get().reconnectObjects(description, _objectDistance * 1.7f);
    }
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

void CreatorWindow::createDisc()
{
    if (_innerRadius > _outerRadius || _innerRadius < 0 || _outerRadius < 0) {
        return;
    }

    ContentDesc description;
    auto const color = EditorModel::get().getDefaultColorCode();
    auto const objectType = isEnergyMaterial() ? ObjectTypeDesc{SolidDesc()} : getObjectTypeDesc();
    auto constexpr SmallValue = 0.01f;
    for (float radius = _innerRadius; radius <= _outerRadius + SmallValue; radius += _objectDistance) {
        float angleInc = [&] {
            if (radius > SmallValue) {
                auto result = asinf(_objectDistance / (2 * radius)) * 2 * toFloat(Const::RadToDeg);
                return 360.0f / floorf(360.0f / result);
            }
            return 360.0f;
        }();
        for (auto angle = 0.0; angle < 360.0f - angleInc / 2; angle += angleInc) {
            auto relPos = Math::unitVectorOfAngle(angle) * radius;
            description._objects.emplace_back(ObjectDesc()
                                                  .id(NumberGenerator::get().createEntityId())
                                                  .stiffness(_stiffness)
                                                  .sticky(_makeSticky)
                                                  .pos(relPos)
                                                  .color(color)
                                                  .isStatic(_static)
                                                  .type(objectType));
        }
    }

    if (isEnergyMaterial()) {
        description = convertToEnergyParticles(description);
    } else {
        DescEditService::get().reconnectObjects(description, _objectDistance * 1.7f);
    }
    DescEditService::get().setCenter(description, getRandomPos());
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

void CreatorWindow::createObjectNetwork(std::vector<RealVector2D> const& positions, float connectionDistance)
{
    if (positions.empty()) {
        return;
    }

    ContentDesc description;
    auto const color = EditorModel::get().getDefaultColorCode();
    auto const objectType = isEnergyMaterial() ? ObjectTypeDesc{SolidDesc()} : getObjectTypeDesc();
    for (auto const& pos : positions) {
        description._objects.emplace_back(ObjectDesc()
                                              .id(NumberGenerator::get().createEntityId())
                                              .stiffness(_stiffness)
                                              .sticky(_makeSticky)
                                              .pos(pos)
                                              .color(color)
                                              .isStatic(_static)
                                              .type(objectType));
    }

    if (isEnergyMaterial()) {
        description = convertToEnergyParticles(description);
    } else {
        DescEditService::get().reconnectObjects(description, connectionDistance);
    }
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

std::vector<RealVector2D> CreatorWindow::calcBezierCurvePath() const
{
    if (_points.size() < 2) {
        return _points;
    }

    auto controlPolygonLength = 0.0f;
    for (auto const& [from, to] : std::views::zip(_points, _points | std::views::drop(1))) {
        controlPolygonLength += Math::length(to - from);
    }

    auto numSegments = std::clamp(toInt(controlPolygonLength * 4 / _objectDistance), 16, 10000);
    std::vector<RealVector2D> result;
    result.reserve(numSegments + 1);
    for (auto segment : std::views::iota(0, numSegments + 1)) {
        result.emplace_back(evaluateBezier(_points, toFloat(segment) / toFloat(numSegments)));
    }
    return result;
}

ContentDesc CreatorWindow::convertToEnergyParticles(ContentDesc const& description) const
{
    ContentDesc result;
    auto const color = EditorModel::get().getDefaultColorCode();
    for (auto const& object : description._objects) {
        result._energies.emplace_back(EnergyDesc().pos(object._pos).energy(_energy).color(color));
    }
    return result;
}

void CreatorWindow::validateAndCorrect()
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

bool CreatorWindow::isEnergyMaterial() const
{
    return _material == CreationMaterial_EnergyParticle;
}

bool CreatorWindow::isPointPlacementMode() const
{
    return _mode == CreationMode_CreateLine || _mode == CreationMode_CreateCurve || _mode == CreationMode_CreatePolygon;
}

InteractionMode CreatorWindow::getInteractionMode() const
{
    if (_mode == CreationMode_Drawing) {
        return InteractionMode_Drawing;
    }
    if (isPointPlacementMode() && _pointPlacementActive) {
        return InteractionMode_PointPlacement;
    }
    return InteractionMode_Selection;
}

ObjectTypeDesc CreatorWindow::getObjectTypeDesc() const
{
    switch (_material) {
    case CreationMaterial_Solid:
        return SolidDesc().energy(_energy);
    case CreationMaterial_Fluid:
        return FluidDesc().energy(_energy).glow(_glow);
    case CreationMaterial_FreeCell:
        return FreeCellDesc().energy(_energy);
    default:
        CHECK(false);
    }
}

RealVector2D CreatorWindow::getRandomPos() const
{
    auto result = Viewport::get().getCenterInWorldPos();
    result.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    result.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    return result;
}
