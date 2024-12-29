#include "CreatorWindow.h"

#include <cstdlib>
#include <cmath>

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"
#include "Fonts/AlienIconFont.h"

#include "Base/NumberGenerator.h"
#include "Base/Math.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"

#include "StyleRepository.h"
#include "AlienImGui.h"
#include "EditorController.h"
#include "Viewport.h"
#include "EditorModel.h"
#include "HelpStrings.h"
#include "SimulationInteractionController.h"

namespace
{
    auto const ModeText = std::unordered_map<CreationMode, std::string>{
        {CreationMode_CreateParticle, "Create a single energy particle"},
        {CreationMode_CreateCell, "Create a single cell"},
        {CreationMode_CreateRectangle, "Create a rectangular cell network"},
        {CreationMode_CreateHexagon, "Create a hexagonal cell network"},
        {CreationMode_CreateDisc, "Create a disc-shaped cell network"},
        {CreationMode_Drawing, "Draw freehand cell network"},
    };

    auto const RightColumnWidth = 160.0f;
}

void CreatorWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void CreatorWindow::processIntern()
{
    AlienImGui::SelectableToolbarButton(ICON_FA_SUN, _mode, CreationMode_CreateParticle, CreationMode_CreateParticle);
    AlienImGui::Tooltip(ModeText.at(CreationMode_CreateParticle));

    ImGui::SameLine();
    AlienImGui::SelectableToolbarButton(ICON_DOT, _mode, CreationMode_CreateCell, CreationMode_CreateCell);
    AlienImGui::Tooltip(ModeText.at(CreationMode_CreateCell));

    ImGui::SameLine();
    AlienImGui::SelectableToolbarButton(ICON_RECTANGLE, _mode, CreationMode_CreateRectangle, CreationMode_CreateRectangle);
    AlienImGui::Tooltip(ModeText.at(CreationMode_CreateRectangle));

    ImGui::SameLine();
    AlienImGui::SelectableToolbarButton(ICON_HEXAGON, _mode, CreationMode_CreateHexagon, CreationMode_CreateHexagon);
    AlienImGui::Tooltip(ModeText.at(CreationMode_CreateHexagon));

    ImGui::SameLine();
    AlienImGui::SelectableToolbarButton(ICON_DISC, _mode, CreationMode_CreateDisc, CreationMode_CreateDisc);
    AlienImGui::Tooltip(ModeText.at(CreationMode_CreateDisc));

    ImGui::SameLine();
    AlienImGui::SelectableToolbarButton(ICON_FA_PAINT_BRUSH, _mode, CreationMode_Drawing, CreationMode_Drawing);
    AlienImGui::Tooltip(ModeText.at(CreationMode_Drawing));

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50.0f)), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        AlienImGui::Group(ModeText.at(_mode));

        auto color = EditorModel::get().getDefaultColorCode();
        AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(RightColumnWidth).tooltip(Const::GenomeColorTooltip), color);
        EditorModel::get().setDefaultColorCode(color);
        if (_mode == CreationMode_Drawing) {
            auto pencilWidth = EditorModel::get().getPencilWidth();
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Pencil radius")
                    .min(1.0f)
                    .max(8.0f)
                    .textWidth(RightColumnWidth)
                    .format("%.1f")
                    .tooltip(Const::CreatorPencilRadiusTooltip),
                &pencilWidth);
            EditorModel::get().setPencilWidth(pencilWidth);
        }
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(RightColumnWidth).tooltip(Const::CellEnergyTooltip), _energy);
        if (_mode != CreationMode_CreateParticle) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters().name("Stiffness").max(1.0f).min(0.0f).textWidth(RightColumnWidth).tooltip(Const::CellStiffnessTooltip),
                &_stiffness);
        }
        
        if (_mode == CreationMode_CreateCell) {
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters()
                    .name("Max connections")
                    .max(MAX_CELL_BONDS)
                    .textWidth(RightColumnWidth)
                    .tooltip(Const::CellMaxConnectionTooltip),
                &_maxConnections);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Set execution order")
                    .textWidth(RightColumnWidth)
                    .tooltip(Const::CreatorAscendingExecutionOrderNumberTooltip),
                _ascendingExecutionNumbers);
        }
        if (_mode == CreationMode_CreateRectangle) {
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters().name("Horizontal cells").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleWidthTooltip),
                _rectHorizontalCells);
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters().name("Vertical cells").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleHeightTooltip),
                _rectVerticalCells);
        }
        if (_mode == CreationMode_CreateHexagon) {
            AlienImGui::InputInt(
                AlienImGui::InputIntParameters().name("Layers").textWidth(RightColumnWidth).tooltip(Const::CreatorHexagonLayersTooltip), _layers);
        }
        if (_mode == CreationMode_CreateDisc) {
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters().name("Outer radius").textWidth(RightColumnWidth).format("%.2f").tooltip(Const::CreatorDiscOuterRadiusTooltip),
                _outerRadius);
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters().name("Inner radius").textWidth(RightColumnWidth).format("%.2f").tooltip(Const::CreatorDiscInnerRadiusTooltip),
                _innerRadius);
        }
        if (_mode == CreationMode_CreateRectangle || _mode == CreationMode_CreateHexagon || _mode == CreationMode_CreateDisc) {
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters()
                    .name("Cell distance")
                    .format("%.2f")
                    .step(0.1)
                    .textWidth(RightColumnWidth)
                    .tooltip(Const::CreatorDistanceTooltip),
                _cellDistance);
        }
        if (_mode != CreationMode_CreateParticle & _mode != CreationMode_CreateCell) {
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Sticky").textWidth(RightColumnWidth).tooltip(Const::CreatorStickyTooltip), _makeSticky);
        }
        if (_mode != CreationMode_CreateParticle) {
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters().name("Indestructible wall").textWidth(RightColumnWidth).tooltip(Const::CellIndestructibleTooltip), _barrier);
        }
    }
    ImGui::EndChild();

    AlienImGui::Separator();
    auto& simInteractionController = SimulationInteractionController::get();
    if (_mode == CreationMode_Drawing) {
        auto text = simInteractionController.isDrawMode() ? "End drawing" : "Start drawing";
        if (AlienImGui::Button(text)) {
            simInteractionController.setDrawMode(!simInteractionController.isDrawMode());
        }
    } else {
        simInteractionController.setDrawMode(false);
        if (AlienImGui::Button("Build")) {
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
    if (!_drawingDataDescription.isEmpty()) {
        _simulationFacade->removeSelectedObjects(false);
    }

    auto createAlignedCircle = [&](auto pos) {
        if (EditorModel::get().getPencilWidth() > 1 + NEAR_ZERO) {
            pos.x = toFloat(toInt(pos.x));
            pos.y = toFloat(toInt(pos.y));
        }
        return DescriptionEditService::get().createUnconnectedCircle(DescriptionEditService::CreateUnconnectedCircleParameters()
                                                              .center(pos)
                                                              .radius(EditorModel::get().getPencilWidth())
                                                              .energy(_energy)
                                                              .stiffness(_stiffness)
                                                              .cellDistance(1.0f)
                                                              .maxConnections(MAX_CELL_BONDS)
                                                              .color(EditorModel::get().getDefaultColorCode())
                                                              .barrier(_barrier));
    };

    if (_drawingDataDescription.isEmpty()) {
        DescriptionEditService::get().addIfSpaceAvailable(_drawingDataDescription, _drawingOccupancy, createAlignedCircle(pos), 0.5f, _simulationFacade->getWorldSize());
        _lastDrawPos = pos;
    } else {
        auto posDelta = Math::length(pos - _lastDrawPos);
        if (posDelta > 0) {
            auto lastDrawPos = _lastDrawPos;
            for (float interDelta = 0; interDelta < posDelta; interDelta += 1.0f) {
                auto drawPos = lastDrawPos + (pos - lastDrawPos) * interDelta / posDelta;
                auto toAdd = createAlignedCircle(drawPos);
                DescriptionEditService::get().addIfSpaceAvailable(_drawingDataDescription, _drawingOccupancy, toAdd, 0.5f, _simulationFacade->getWorldSize());
                _lastDrawPos = drawPos;
            }
        }
    }
    DescriptionEditService::get().reconnectCells(_drawingDataDescription, 1.5f);
    if (!_makeSticky) {
        auto origDrawing = _drawingDataDescription;
        DescriptionEditService::get().removeStickiness(_drawingDataDescription);
        _simulationFacade->addAndSelectSimulationData(_drawingDataDescription);
        _drawingDataDescription = origDrawing;
    } else {
        _simulationFacade->addAndSelectSimulationData(_drawingDataDescription);
    }

    _simulationFacade->reconnectSelectedObjects();
    EditorModel::get().update();
}

void CreatorWindow::finishDrawing()
{
    _drawingDataDescription.clear();
    _drawingOccupancy.clear();
}

CreatorWindow::CreatorWindow()
    : AlienWindow("Creator", "editors.creator", false)
{}

void CreatorWindow::createCell()
{
    auto creatureId = toInt(NumberGenerator::get().getRandomInt(std::numeric_limits<int>::max()));

    auto cell = CellDescription()
                    .setPos(getRandomPos())
                    .setEnergy(_energy)
                    .setStiffness(_stiffness)
                    .setMaxConnections(_maxConnections)
                    .setExecutionOrderNumber(_lastExecutionNumber)
                    .setColor(EditorModel::get().getDefaultColorCode())
                    .setBarrier(_barrier)
                    .setCreatureId(creatureId);
    if (_ascendingExecutionNumbers) {
        cell.setInputExecutionOrderNumber((_lastExecutionNumber + 5) % 6);
    }
    auto data = DataDescription().addCell(cell);
    _simulationFacade->addAndSelectSimulationData(data);
    incExecutionNumber();
}

void CreatorWindow::createParticle()
{
    auto particle = ParticleDescription().setPos(getRandomPos()).setEnergy(_energy);
    auto data = DataDescription().addParticle(particle);
    _simulationFacade->addAndSelectSimulationData(data);
}

void CreatorWindow::createRectangle()
{
    if (_rectHorizontalCells <= 0 || _rectVerticalCells <= 0) {
        return;
    }

    auto data = DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                  .width(_rectHorizontalCells)
                                                  .height(_rectVerticalCells)
                                                  .cellDistance(_cellDistance)
                                                  .energy(_energy)
                                                  .stiffness(_stiffness)
                                                  .removeStickiness(!_makeSticky)
                                                  .maxConnections(MAX_CELL_BONDS)
                                                  .color(EditorModel::get().getDefaultColorCode())
                                                  .center(getRandomPos())
                                                  .barrier(_barrier));

    _simulationFacade->addAndSelectSimulationData(data);
}

void CreatorWindow::createHexagon()
{
    if (_layers <= 0) {
        return;
    }
    DataDescription data = DescriptionEditService::get().createHex(DescriptionEditService::CreateHexParameters()
                                                            .layers(_layers)
                                                            .cellDistance(_cellDistance)
                                                            .energy(_energy)
                                                            .stiffness(_stiffness)
                                                            .removeStickiness(!_makeSticky)
                                                            .maxConnections(MAX_CELL_BONDS)
                                                            .color(EditorModel::get().getDefaultColorCode())
                                                            .center(getRandomPos())
                                                            .barrier(_barrier));
    _simulationFacade->addAndSelectSimulationData(data);
}

void CreatorWindow::createDisc()
{
    if (_innerRadius > _outerRadius || _innerRadius < 0 || _outerRadius <= 0) {
        return;
    }

    DataDescription data;
    auto constexpr SmallValue = 0.01f;
    for (float radius = _innerRadius; radius - SmallValue <= _outerRadius; radius += _cellDistance) {
        float angleInc =
            [&] {
                if (radius > SmallValue) {
                    auto result = asinf(_cellDistance / (2 * radius)) * 2 * toFloat(Const::RadToDeg);
                    return 360.0f / floorf(360.0f / result);
                }
                return 360.0f;
            }();
        std::unordered_set<uint64_t> cellIds;
        for (auto angle = 0.0; angle < 360.0f - angleInc / 2; angle += angleInc) {
            auto relPos = Math::unitVectorOfAngle(angle) * radius;

            data.addCell(CellDescription()
                             .setId(NumberGenerator::get().getId())
                             .setEnergy(_energy)
                             .setStiffness(_stiffness)
                             .setPos(relPos)
                             .setMaxConnections(MAX_CELL_BONDS)
                             .setColor(EditorModel::get().getDefaultColorCode())
                             .setBarrier(_barrier));
        }
    }

    DescriptionEditService::get().reconnectCells(data, _cellDistance * 1.7f);
    if (!_makeSticky) {
        DescriptionEditService::get().removeStickiness(data);
    }
    data.setCenter(getRandomPos());
    _simulationFacade->addAndSelectSimulationData(data);
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

void CreatorWindow::incExecutionNumber()
{
    if (_ascendingExecutionNumbers) {
        auto parameters = _simulationFacade->getSimulationParameters();
        _lastExecutionNumber = (_lastExecutionNumber + 1) % parameters.cellNumExecutionOrderNumbers;
    }
}
