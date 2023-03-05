#include "CreatorWindow.h"

#include <cstdlib>
#include <cmath>

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"
#include "Fonts/AlienIconFont.h"

#include "Base/NumberGenerator.h"
#include "Base/Math.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"

#include "GlobalSettings.h"
#include "StyleRepository.h"
#include "AlienImGui.h"
#include "Viewport.h"
#include "EditorModel.h"

namespace
{
    auto const ModeText = std::unordered_map<CreationMode, std::string>{
        {CreationMode::CreateParticle, "Create single energy particle"},
        {CreationMode::CreateCell, "Create single cell"},
        {CreationMode::CreateRectangle, "Create rectangular cell cluster"},
        {CreationMode::CreateHexagon, "Create hexagonal cell cluster"},
        {CreationMode::CreateDisc, "Create disc-shaped cell cluster"},
        {CreationMode::Drawing, "Draw freehand cell cluster"},
    };

    auto const RightColumnWidth = 180.0f;
}

_CreatorWindow::_CreatorWindow(EditorModel const& editorModel, SimulationController const& simController, Viewport const& viewport)
    : _AlienWindow("Creator", "editor.creator", true), _editorModel(editorModel)
    , _simController(simController)
    , _viewport(viewport)
{
}

void _CreatorWindow::processIntern()
{
    if (AlienImGui::ToolbarButton(ICON_FA_SUN)) {
        _mode = CreationMode::CreateParticle;
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_DOT)) {
        _mode = CreationMode::CreateCell;
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_RECTANGLE)) {
        _mode = CreationMode::CreateRectangle;
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_HEXAGON)) {
        _mode = CreationMode::CreateHexagon;
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_DISC)) {
        _mode = CreationMode::CreateDisc;
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_PAINT_BRUSH)) {
        _mode = CreationMode::Drawing;
    }

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - contentScale(50.0f)), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        AlienImGui::Group(ModeText.at(_mode));

        auto color = _editorModel->getDefaultColorCode();
        AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(RightColumnWidth), color);
        _editorModel->setDefaultColorCode(color);
        if (_mode == CreationMode::Drawing) {
            auto pencilWidth = _editorModel->getPencilWidth();
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters().name("Pencil width").min(1.0f).max(8.0f).textWidth(RightColumnWidth).format("%.1f"), pencilWidth);
            _editorModel->setPencilWidth(pencilWidth);
        }
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(RightColumnWidth), _energy);
        if (_mode != CreationMode::CreateParticle) {
            AlienImGui::SliderFloat(AlienImGui::SliderFloatParameters().name("Stiffness").max(1.0f).min(0.0f).textWidth(RightColumnWidth), _stiffness);
        }
        
        if (_mode == CreationMode::CreateCell) {
            AlienImGui::SliderInt(AlienImGui::SliderIntParameters().name("Max connections").max(MAX_CELL_BONDS).textWidth(RightColumnWidth), &_maxConnections);
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Ascending execution order").textWidth(RightColumnWidth), _ascendingExecutionNumbers);
        }
        if (_mode == CreationMode::CreateRectangle) {
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Horizontal cells").textWidth(RightColumnWidth), _rectHorizontalCells);
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Vertical cells").textWidth(RightColumnWidth), _rectVerticalCells);
        }
        if (_mode == CreationMode::CreateHexagon) {
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Layers").textWidth(RightColumnWidth), _layers);
        }
        if (_mode == CreationMode::CreateDisc) {
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Outer radius").textWidth(RightColumnWidth).format("%.2f"), _outerRadius);
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Inner radius").textWidth(RightColumnWidth).format("%.2f"), _innerRadius);
        }
        if (_mode == CreationMode::CreateRectangle || _mode == CreationMode::CreateHexagon || _mode == CreationMode::CreateDisc) {
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters().name("Cell distance").format("%.2f").step(0.1).textWidth(RightColumnWidth), _cellDistance);
        }
        if (_mode != CreationMode::CreateParticle & _mode != CreationMode::CreateCell) {
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Make sticky").textWidth(RightColumnWidth), _makeSticky);
        }
        AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Attach to background").textWidth(RightColumnWidth), _barrier);
    }
    ImGui::EndChild();

    AlienImGui::Separator();
    if (_mode == CreationMode::Drawing) {
        auto text = _editorModel->isDrawMode() ? "End drawing" : "Start drawing";
        if (AlienImGui::Button(text)) {
            _editorModel->setDrawMode(!_editorModel->isDrawMode());
        }
    } else {
        _editorModel->setDrawMode(false);
        if (AlienImGui::Button("Build")) {
            if (_mode == CreationMode::CreateCell) {
                createCell();
            }
            if (_mode == CreationMode::CreateParticle) {
                createParticle();
            }
            if (_mode == CreationMode::CreateRectangle) {
                createRectangle();
            }
            if (_mode == CreationMode::CreateHexagon) {
                createHexagon();
            }
            if (_mode == CreationMode::CreateDisc) {
                createDisc();
            }
            _editorModel->update();
        }
    }
    validationAndCorrection();
}

void _CreatorWindow::onDrawing()
{
    auto mousePos = ImGui::GetMousePos();
    auto pos = _viewport->mapViewToWorldPosition({mousePos.x, mousePos.y});
    if (!_drawing.isEmpty()) {
        _simController->removeSelectedObjects(false);
    }

    auto createAlignedCircle = [&](auto pos) {
        if (_editorModel->getPencilWidth() > 1 + NEAR_ZERO) {
            pos.x = toFloat(toInt(pos.x));
            pos.y = toFloat(toInt(pos.y));
        }
        return DescriptionHelper::createUnconnectedCircle(DescriptionHelper::CreateUnconnectedCircleParameters()
                                                              .center(pos)
                                                              .radius(_editorModel->getPencilWidth())
                                                              .energy(_energy)
                                                              .stiffness(_stiffness)
                                                              .cellDistance(1.0f)
                                                              .maxConnections(MAX_CELL_BONDS)
                                                              .color(_editorModel->getDefaultColorCode())
                                                              .barrier(_barrier));
    };

    if (_drawing.isEmpty()) {
        DescriptionHelper::addIfSpaceAvailable(_drawing, _drawingOccupancy, createAlignedCircle(pos), 0.5f, _simController->getWorldSize());
        _lastDrawPos = pos;
    } else {
        auto posDelta = Math::length(pos - _lastDrawPos);
        if (posDelta > 0) {
            auto lastDrawPos = _lastDrawPos;
            for (float interDelta = 0; interDelta < posDelta; interDelta += 1.0f) {
                auto drawPos = lastDrawPos + (pos - lastDrawPos) * interDelta / posDelta;
                auto toAdd = createAlignedCircle(drawPos);
                DescriptionHelper::addIfSpaceAvailable(_drawing, _drawingOccupancy, toAdd, 0.5f, _simController->getWorldSize());
                _lastDrawPos = drawPos;
            }
        }
    }
    DescriptionHelper::reconnectCells(_drawing, 1.5f);
    if (!_makeSticky) {
        auto origDrawing = _drawing;
        DescriptionHelper::removeStickiness(_drawing);
        _simController->addAndSelectSimulationData(_drawing);
        _drawing = origDrawing;
    } else {
        _simController->addAndSelectSimulationData(_drawing);
    }

    _simController->reconnectSelectedObjects();
    _editorModel->update();
}

void _CreatorWindow::finishDrawing()
{
    _drawing.clear();
    _drawingOccupancy.clear();
}

void _CreatorWindow::createCell()
{
    auto cell = CellDescription()
                    .setPos(getRandomPos())
                    .setEnergy(_energy)
                    .setStiffness(_stiffness)
                    .setMaxConnections(_maxConnections)
                    .setExecutionOrderNumber(_lastExecutionNumber)
                    .setColor(_editorModel->getDefaultColorCode())
                    .setBarrier(_barrier);
    auto data = DataDescription().addCell(cell);
    _simController->addAndSelectSimulationData(data);
    incExecutionNumber();
}

void _CreatorWindow::createParticle()
{
    auto particle = ParticleDescription().setPos(getRandomPos()).setEnergy(_energy);
    auto data = DataDescription().addParticle(particle);
    _simController->addAndSelectSimulationData(data);
}

void _CreatorWindow::createRectangle()
{
    if (_rectHorizontalCells <= 0 || _rectVerticalCells <= 0) {
        return;
    }

    auto parameters = _simController->getSimulationParameters();
    auto data = DescriptionHelper::createRect(DescriptionHelper::CreateRectParameters()
                                                  .width(_rectHorizontalCells)
                                                  .height(_rectVerticalCells)
                                                  .cellDistance(_cellDistance)
                                                  .energy(_energy)
                                                  .stiffness(_stiffness)
                                                  .removeStickiness(!_makeSticky)
                                                  .maxConnections(MAX_CELL_BONDS)
                                                  .color(_editorModel->getDefaultColorCode())
                                                  .center(getRandomPos())
                                                  .barrier(_barrier));

    _simController->addAndSelectSimulationData(data);
}

void _CreatorWindow::createHexagon()
{
    if (_layers <= 0) {
        return;
    }
    DataDescription data = DescriptionHelper::createHex(DescriptionHelper::CreateHexParameters()
                                                            .layers(_layers)
                                                            .cellDistance(_cellDistance)
                                                            .energy(_energy)
                                                            .stiffness(_stiffness)
                                                            .removeStickiness(!_makeSticky)
                                                            .maxConnections(MAX_CELL_BONDS)
                                                            .color(_editorModel->getDefaultColorCode())
                                                            .center(getRandomPos())
                                                            .barrier(_barrier));
    _simController->addAndSelectSimulationData(data);
}

void _CreatorWindow::createDisc()
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
                             .setId(NumberGenerator::getInstance().getId())
                             .setEnergy(_energy)
                             .setStiffness(_stiffness)
                             .setPos(relPos)
                             .setMaxConnections(MAX_CELL_BONDS)
                             .setColor(_editorModel->getDefaultColorCode())
                             .setBarrier(_barrier));
        }
    }

    DescriptionHelper::reconnectCells(data, _cellDistance * 1.7f);
    if (!_makeSticky) {
        DescriptionHelper::removeStickiness(data);
    }
    data.setCenter(getRandomPos());
    _simController->addAndSelectSimulationData(data);
}

void _CreatorWindow::validationAndCorrection()
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

RealVector2D _CreatorWindow::getRandomPos() const
{
    auto result = _viewport->getCenterInWorldPos();
    result.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    result.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    return result;
}

void _CreatorWindow::incExecutionNumber()
{
    if (_ascendingExecutionNumbers) {
        auto parameters = _simController->getSimulationParameters();
        _lastExecutionNumber = (_lastExecutionNumber + 1) % parameters.cellNumExecutionOrderNumbers;
    }
}
