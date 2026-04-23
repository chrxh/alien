#include "CreatorWindow.h"

#include <cmath>
#include <cstdlib>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "EditorController.h"
#include "EditorModel.h"
#include "HelpStrings.h"
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
        {CreationMode_Drawing, "Draw freehand"},
    };

    auto const RightColumnWidth = 160.0f;
}

void CreatorWindow::initIntern() {}

void CreatorWindow::processIntern()
{
    AlienGui::SelectableToolbarButton(ICON_DOT, _mode, CreationMode_CreateObject, CreationMode_CreateObject);
    AlienGui::Tooltip(ModeText.at(CreationMode_CreateObject));

    ImGui::SameLine();
    AlienGui::SelectableToolbarButton(ICON_RECTANGLE, _mode, CreationMode_CreateRectangle, CreationMode_CreateRectangle);
    AlienGui::Tooltip(ModeText.at(CreationMode_CreateRectangle));

    ImGui::SameLine();
    AlienGui::SelectableToolbarButton(ICON_HEXAGON, _mode, CreationMode_CreateHexagon, CreationMode_CreateHexagon);
    AlienGui::Tooltip(ModeText.at(CreationMode_CreateHexagon));

    ImGui::SameLine();
    AlienGui::SelectableToolbarButton(ICON_DISC, _mode, CreationMode_CreateDisc, CreationMode_CreateDisc);
    AlienGui::Tooltip(ModeText.at(CreationMode_CreateDisc));

    ImGui::SameLine();
    AlienGui::SelectableToolbarButton(ICON_FA_PAINT_BRUSH, _mode, CreationMode_Drawing, CreationMode_Drawing);
    AlienGui::Tooltip(ModeText.at(CreationMode_Drawing));

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50.0f)), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        AlienGui::Group(AlienGui::GroupParameters().text(ModeText.at(_mode)));

        auto color = EditorModel::get().getDefaultColorCode();
        AlienGui::ComboColor(AlienGui::ComboColorParameters().name("Color").textWidth(RightColumnWidth).tooltip(Const::GenomeColorTooltip), color);
        EditorModel::get().setDefaultColorCode(color);
        if (_mode == CreationMode_Drawing) {
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
        int material = _material;
        AlienGui::Switcher(
            AlienGui::SwitcherParameters()
                .name("Material")
                .textWidth(RightColumnWidth)
                .values({"Solid", "Fluid", "Free cells", "Energy particles"})
                .tooltip(Const::CreatorDrawingTypeTooltip),
            material);
        _material = material;
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(RightColumnWidth).tooltip(Const::CellEnergyTooltip), _energy);
        if (_material == CreationMaterial_Fluid) {
            AlienGui::SliderFloat(AlienGui::SliderFloatParameters().name("Glow").min(0).max(1.0f).format("%.2f").textWidth(RightColumnWidth), &_glow);
        }
        if (!isEnergyMaterial() && _material != CreationMaterial_Fluid) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Stiffness").max(1.0f).min(0.0f).textWidth(RightColumnWidth).tooltip(Const::CellStiffnessTooltip),
                &_stiffness);
        }

        if (_mode == CreationMode_CreateRectangle) {
            AlienGui::InputInt(
                AlienGui::InputIntParameters().name("Horizontal objects").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleWidthTooltip),
                _rectHorizontalObjects);
            AlienGui::InputInt(
                AlienGui::InputIntParameters().name("Vertical objects").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleHeightTooltip),
                _rectVerticalObjects);
        }
        if (_mode == CreationMode_CreateHexagon) {
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Layers").textWidth(RightColumnWidth).tooltip(Const::CreatorHexagonLayersTooltip), _layers);
        }
        if (_mode == CreationMode_CreateDisc) {
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters().name("Outer radius").textWidth(RightColumnWidth).format("%.0f").tooltip(Const::CreatorDiscOuterRadiusTooltip),
                _outerRadius);
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters().name("Inner radius").textWidth(RightColumnWidth).format("%.0f").tooltip(Const::CreatorDiscInnerRadiusTooltip),
                _innerRadius);
        }
        if (_mode == CreationMode_CreateRectangle || _mode == CreationMode_CreateHexagon || _mode == CreationMode_CreateDisc) {
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters()
                    .name("Object distance")
                    .format("%.2f")
                    .step(0.1)
                    .textWidth(RightColumnWidth)
                    .tooltip(Const::CreatorDistanceTooltip),
                _objectDistance);
        }
        if (_mode != CreationMode_CreateObject) {
            AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Sticky").textWidth(RightColumnWidth).tooltip(Const::CreatorStickyTooltip), _makeSticky);
        }
        if (!isEnergyMaterial()) {
            AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Fixed").textWidth(RightColumnWidth).tooltip(Const::CellFixedTooltip), _fixed);
        }
    }
    ImGui::EndChild();

    auto& simInteractionController = SimulationInteractionController::get();
    if (_mode == CreationMode_Drawing) {
        simInteractionController.setDrawMode(true);
    } else {
        AlienGui::Separator();
        simInteractionController.setDrawMode(false);
        if (AlienGui::Button("Build")) {
            if (_mode == CreationMode_CreateObject) {
                createEntity();
            }
            if (_mode == CreationMode_CreateRectangle) {
                createRectangle();
            }
            if (_mode == CreationMode_CreateHexagon) {
                createHexagon();
            }
            if (_mode == CreationMode_CreateDisc) {
                createDisc();
            }
            EditorModel::get().update();
        }
    }
    validateAndCorrect();
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
        if (isEnergyMaterial()) {
            auto circle = DescEditService::get().createCircle(
                DescEditService::CreateCircleParameters().center(pos).radius(EditorModel::get().getPencilWidth()).connectObjects(false));
            Desc result;
            for (auto const& object : circle._objects) {
                result._energies.emplace_back(EnergyDesc().pos(object._pos).energy(_energy).color(EditorModel::get().getDefaultColorCode()));
            }
            return result;
        }
        return DescEditService::get().createCircle(DescEditService::CreateCircleParameters()
                                                       .center(pos)
                                                       .radius(EditorModel::get().getPencilWidth())
                                                       .type(getObjectTypeDesc())
                                                       .stiffness(_stiffness)
                                                       .sticky(_makeSticky)
                                                       .cellDistance(1.0f)
                                                       .color(EditorModel::get().getDefaultColorCode())
                                                       .fixed(_fixed)
                                                       .connectObjects(false));
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
        Desc newEntities;
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

CreatorWindow::CreatorWindow()
    : AlienWindow("Creator", "editors.creator", false)
{}

void CreatorWindow::createEntity()
{
    Desc description;
    if (isEnergyMaterial()) {
        description._energies.emplace_back(EnergyDesc().pos(getRandomPos()).energy(_energy).color(EditorModel::get().getDefaultColorCode()));
    } else {
        description._objects.emplace_back(ObjectDesc()
                                              .pos(getRandomPos())
                                              .stiffness(_stiffness)
                                              .color(EditorModel::get().getDefaultColorCode())
                                              .fixed(_fixed)
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

    if (isEnergyMaterial()) {
        Desc description;
        auto center = getRandomPos();
        auto color = EditorModel::get().getDefaultColorCode();
        auto horizontalOffset = toFloat(_rectHorizontalObjects - 1) / 2.0f;
        auto verticalOffset = toFloat(_rectVerticalObjects - 1) / 2.0f;
        for (int y = 0; y < _rectVerticalObjects; ++y) {
            for (int x = 0; x < _rectHorizontalObjects; ++x) {
                auto pos =
                    RealVector2D{center.x + (toFloat(x) - horizontalOffset) * _objectDistance, center.y + (toFloat(y) - verticalOffset) * _objectDistance};
                description._energies.emplace_back(EnergyDesc().pos(pos).energy(_energy).color(color));
            }
        }
        _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
        return;
    }

    auto description = DescEditService::get().createRect(DescEditService::CreateRectParameters()
                                                             .objectType(getObjectTypeDesc())
                                                             .width(_rectHorizontalObjects)
                                                             .height(_rectVerticalObjects)
                                                             .cellDistance(_objectDistance)
                                                             .stiffness(_stiffness)
                                                             .sticky(_makeSticky)
                                                             .color(EditorModel::get().getDefaultColorCode())
                                                             .center(getRandomPos())
                                                             .fixed(_fixed));

    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

void CreatorWindow::createHexagon()
{
    if (_layers <= 0) {
        return;
    }
    if (isEnergyMaterial()) {
        auto objectTemplate = DescEditService::get().createHex(
            DescEditService::CreateHexParameters().objectType(SolidDesc()).layers(_layers).cellDistance(_objectDistance).connectObjects(false));
        Desc description;
        auto color = EditorModel::get().getDefaultColorCode();
        for (auto const& object : objectTemplate._objects) {
            description._energies.emplace_back(EnergyDesc().pos(object._pos).energy(_energy).color(color));
        }
        DescEditService::get().setCenter(description, getRandomPos());
        _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
        return;
    }

    Desc description = DescEditService::get().createHex(DescEditService::CreateHexParameters()
                                                            .objectType(getObjectTypeDesc())
                                                            .layers(_layers)
                                                            .cellDistance(_objectDistance)
                                                            .stiffness(_stiffness)
                                                            .sticky(_makeSticky)
                                                            .color(EditorModel::get().getDefaultColorCode())
                                                            .center(getRandomPos())
                                                            .fixed(_fixed));
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
    DescEditService::get().reconnectObjects(description, _objectDistance * 1.7f);
}

void CreatorWindow::createDisc()
{
    if (_innerRadius > _outerRadius || _innerRadius < 0 || _outerRadius < 0) {
        return;
    }

    Desc description;
    auto color = EditorModel::get().getDefaultColorCode();
    auto constexpr SmallValue = 0.01f;
    for (float radius = _innerRadius; radius <= _outerRadius + SmallValue; radius += _objectDistance) {
        float angleInc = [&] {
            if (radius > SmallValue) {
                auto result = asinf(_objectDistance / (2 * radius)) * 2 * toFloat(Const::RadToDeg);
                return 360.0f / floorf(360.0f / result);
            }
            return 360.0f;
        }();
        std::unordered_set<uint64_t> objectIds;
        for (auto angle = 0.0; angle < 360.0f - angleInc / 2; angle += angleInc) {
            auto relPos = Math::unitVectorOfAngle(angle) * radius;
            if (isEnergyMaterial()) {
                description._energies.emplace_back(EnergyDesc().pos(relPos).energy(_energy).color(color));
            } else {
                description._objects.emplace_back(ObjectDesc()
                                                      .id(NumberGenerator::get().createEntityId())
                                                      .stiffness(_stiffness)
                                                      .sticky(_makeSticky)
                                                      .pos(relPos)
                                                      .color(color)
                                                      .fixed(_fixed)
                                                      .type(getObjectTypeDesc()));
            }
        }
    }

    if (!isEnergyMaterial()) {
        DescEditService::get().reconnectObjects(description, _objectDistance * 1.7f);
    }
    DescEditService::get().setCenter(description, getRandomPos());
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

void CreatorWindow::validateAndCorrect()
{
    _energy = std::max(0.0f, _energy);
    _stiffness = std::min(1.0f, std::max(0.0f, _stiffness));
    _material = std::max(static_cast<int>(CreationMaterial_Solid), std::min(static_cast<int>(CreationMaterial_EnergyParticle), _material));
    _objectDistance = std::min(10.0f, std::max(0.1f, _objectDistance));
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
