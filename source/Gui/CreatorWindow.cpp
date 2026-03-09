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
        {CreationMode_CreateParticle, "Create a single energy particle"},
        {CreationMode_CreateCell, "Create a single cell"},
        {CreationMode_CreateRectangle, "Create a rectangular cell network"},
        {CreationMode_CreateHexagon, "Create a hexagonal cell network"},
        {CreationMode_CreateDisc, "Create a disc-shaped cell network"},
        {CreationMode_Drawing, "Draw freehand"},
    };

    auto const RightColumnWidth = 160.0f;
}

void CreatorWindow::initIntern() {}

void CreatorWindow::processIntern()
{
    AlienGui::SelectableToolbarButton(ICON_FA_SUN, _mode, CreationMode_CreateParticle, CreationMode_CreateParticle);
    AlienGui::Tooltip(ModeText.at(CreationMode_CreateParticle));

    ImGui::SameLine();
    AlienGui::SelectableToolbarButton(ICON_DOT, _mode, CreationMode_CreateCell, CreationMode_CreateCell);
    AlienGui::Tooltip(ModeText.at(CreationMode_CreateCell));

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
            AlienGui::Switcher(
                AlienGui::SwitcherParameters()
                    .name("Material type")
                    .textWidth(RightColumnWidth)
                    .values({"Solid", "Fluid"})
                    .tooltip(Const::CreatorDrawingTypeTooltip),
                _drawingType);
        }
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(RightColumnWidth).tooltip(Const::CellEnergyTooltip), _energy);
        if (_mode != CreationMode_CreateParticle) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Stiffness").max(1.0f).min(0.0f).textWidth(RightColumnWidth).tooltip(Const::CellStiffnessTooltip),
                &_stiffness);
        }

        if (_mode == CreationMode_CreateRectangle) {
            AlienGui::InputInt(
                AlienGui::InputIntParameters().name("Horizontal cells").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleWidthTooltip),
                _rectHorizontalCells);
            AlienGui::InputInt(
                AlienGui::InputIntParameters().name("Vertical cells").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleHeightTooltip),
                _rectVerticalCells);
        }
        if (_mode == CreationMode_CreateHexagon) {
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Layers").textWidth(RightColumnWidth).tooltip(Const::CreatorHexagonLayersTooltip), _layers);
        }
        if (_mode == CreationMode_CreateDisc) {
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters().name("Outer radius").textWidth(RightColumnWidth).format("%.2f").tooltip(Const::CreatorDiscOuterRadiusTooltip),
                _outerRadius);
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters().name("Inner radius").textWidth(RightColumnWidth).format("%.2f").tooltip(Const::CreatorDiscInnerRadiusTooltip),
                _innerRadius);
        }
        if (_mode == CreationMode_CreateRectangle || _mode == CreationMode_CreateHexagon || _mode == CreationMode_CreateDisc) {
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters()
                    .name("Cell distance")
                    .format("%.2f")
                    .step(0.1)
                    .textWidth(RightColumnWidth)
                    .tooltip(Const::CreatorDistanceTooltip),
                _cellDistance);
        }
        if (_mode != CreationMode_CreateParticle & _mode != CreationMode_CreateCell) {
            AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Sticky").textWidth(RightColumnWidth).tooltip(Const::CreatorStickyTooltip), _makeSticky);
        }
        if (_mode != CreationMode_CreateParticle) {
            AlienGui::Checkbox(
                AlienGui::CheckboxParameters().name("Fixed").textWidth(RightColumnWidth).tooltip(Const::CellFixedTooltip), _fixed);
        }
    }
    ImGui::EndChild();

    AlienGui::Separator();
    auto& simInteractionController = SimulationInteractionController::get();
    if (_mode == CreationMode_Drawing) {
        auto text = simInteractionController.isDrawMode() ? "End drawing" : "Start drawing";
        if (AlienGui::Button(text)) {
            simInteractionController.setDrawMode(!simInteractionController.isDrawMode());
        }
    } else {
        simInteractionController.setDrawMode(false);
        if (AlienGui::Button("Build")) {
            if (_mode == CreationMode_CreateCell) {
                createCell();
            }
            if (_mode == CreationMode_CreateParticle) {
                createParticle();
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
    if (!_drawingDescription.isEmpty()) {
        _SimulationFacade::get()->removeSelectedObjects(false);
    }

    auto createAlignedCircle = [&](auto pos) {
        if (EditorModel::get().getPencilWidth() > 1 + NEAR_ZERO) {
            pos.x = toFloat(toInt(pos.x));
            pos.y = toFloat(toInt(pos.y));
        }
        return DescEditService::get().createUnconnectedCircle(DescEditService::CreateUnconnectedCircleParameters()
                                                                         .center(pos)
                                                                         .radius(EditorModel::get().getPencilWidth())
                                                                         .usableEnergy(_energy)
                                                                         .stiffness(_stiffness)
                                                                         .sticky(_makeSticky)
                                                                         .cellDistance(1.0f)
                                                                         .color(EditorModel::get().getDefaultColorCode())
                                                                         .fixed(_fixed));
    };

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
                DescEditService::get().addIfSpaceAvailable(
                    _drawingDescription, _drawingOccupancy, toAdd, 0.5f, _SimulationFacade::get()->getWorldSize());
                _lastDrawPos = drawPos;
            }
        }
    }
    if (_drawingType == DrawingType_Solid) {
        DescEditService::get().reconnectCells(_drawingDescription, 1.5f);
    }
    _SimulationFacade::get()->addAndSelectSimulationData(Desc(_drawingDescription));

    if (_drawingType == DrawingType_Solid) {
        _SimulationFacade::get()->reconnectSelectedObjects();
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

void CreatorWindow::createCell()
{
    auto object = ObjectDesc().pos(getRandomPos()).stiffness(_stiffness).color(EditorModel::get().getDefaultColorCode()).fixed(_fixed).sticky(_makeSticky).type(StructureDesc());
    Desc description;
    description._objects.emplace_back(object);
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

void CreatorWindow::createParticle()
{
    auto energyParticle = EnergyDesc().pos(getRandomPos()).energy(_energy);
    Desc description;
    description._energies.emplace_back(energyParticle);
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

void CreatorWindow::createRectangle()
{
    if (_rectHorizontalCells <= 0 || _rectVerticalCells <= 0) {
        return;
    }

    auto description = DescEditService::get().createRect(DescEditService::CreateRectParameters()
                                                                    .objectType(StructureDesc())
                                                                    .width(_rectHorizontalCells)
                                                                    .height(_rectVerticalCells)
                                                                    .cellDistance(_cellDistance)
                                                                    .usableEnergy(_energy)
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
    Desc description = DescEditService::get().createHex(DescEditService::CreateHexParameters()
                                                                          .objectType(StructureDesc())
                                                                          .layers(_layers)
                                                                          .cellDistance(_cellDistance)
                                                                          .usableEnergy(_energy)
                                                                          .stiffness(_stiffness)
                                                                          .sticky(_makeSticky)
                                                                          .color(EditorModel::get().getDefaultColorCode())
                                                                          .center(getRandomPos())
                                                                          .fixed(_fixed));
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

void CreatorWindow::createDisc()
{
    if (_innerRadius > _outerRadius || _innerRadius < 0 || _outerRadius <= 0) {
        return;
    }

    Desc description;
    auto constexpr SmallValue = 0.01f;
    for (float radius = _innerRadius; radius - SmallValue <= _outerRadius; radius += _cellDistance) {
        float angleInc = [&] {
            if (radius > SmallValue) {
                auto result = asinf(_cellDistance / (2 * radius)) * 2 * toFloat(Const::RadToDeg);
                return 360.0f / floorf(360.0f / result);
            }
            return 360.0f;
        }();
        std::unordered_set<uint64_t> cellIds;
        for (auto angle = 0.0; angle < 360.0f - angleInc / 2; angle += angleInc) {
            auto relPos = Math::unitVectorOfAngle(angle) * radius;

            description._objects.emplace_back(ObjectDesc().id(NumberGenerator::get().createEntityId()).stiffness(_stiffness).sticky(_makeSticky).pos(relPos).color(EditorModel::get().getDefaultColorCode()).fixed(_fixed).type(StructureDesc()));
        }
    }

    DescEditService::get().reconnectCells(description, _cellDistance * 1.7f);
    DescEditService::get().setCenter(description, getRandomPos());
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(description));
}

void CreatorWindow::validateAndCorrect()
{
    _energy = std::max(0.0f, _energy);
    _stiffness = std::min(1.0f, std::max(0.0f, _stiffness));
    _cellDistance = std::min(10.0f, std::max(0.1f, _cellDistance));
    _rectHorizontalCells = std::max(1, _rectHorizontalCells);
    _rectVerticalCells = std::max(1, _rectVerticalCells);
    _layers = std::max(1, _layers);
    _outerRadius = std::max(_innerRadius, _outerRadius);
    _innerRadius = std::max(1.0f, _innerRadius);
}

RealVector2D CreatorWindow::getRandomPos() const
{
    auto result = Viewport::get().getCenterInWorldPos();
    result.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    result.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    return result;
}
