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
        {CreationMode::CreateParticle, "Create single particle"},
        {CreationMode::CreateCell, "Create single cell"},
        {CreationMode::CreateRectangle, "Create rectangular cell cluster"},
        {CreationMode::CreateHexagon, "Create hexagonal cell cluster"},
        {CreationMode::CreateDisc, "Create disc-shaped cell cluster"},
        {CreationMode::Drawing, "Draw freehand cell cluster"},
    };

    auto const MaxContentTextWidth = 180.0f;
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

    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {

        AlienImGui::Group(ModeText.at(_mode));
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(MaxContentTextWidth), _energy);

        auto parameters = _simController->getSimulationParameters();
        if (_mode == CreationMode::CreateCell) {
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters().name("Max connections").max(parameters.cellMaxBonds).textWidth(MaxContentTextWidth), _maxConnections);
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Ascending branch number").textWidth(MaxContentTextWidth), _ascendingBranchNumbers);
        }
        if (_mode == CreationMode::CreateRectangle) {
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Horizontal cells").textWidth(MaxContentTextWidth), _rectHorizontalCells);
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Vertical cells").textWidth(MaxContentTextWidth), _rectVerticalCells);
        }
        if (_mode == CreationMode::CreateHexagon) {
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Layers").textWidth(MaxContentTextWidth), _layers);
        }
        if (_mode == CreationMode::CreateDisc) {
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Outer radius").textWidth(MaxContentTextWidth).format("%.2f"), _outerRadius);
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Inner radius").textWidth(MaxContentTextWidth).format("%.2f"), _innerRadius);
        }

        if (_mode == CreationMode::CreateRectangle || _mode == CreationMode::CreateHexagon || _mode == CreationMode::CreateDisc
            || _mode == CreationMode::Drawing) {
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters().name("Cell distance").format("%.2f").step(0.1).textWidth(MaxContentTextWidth), _cellDistance);
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Make sticky").textWidth(MaxContentTextWidth), _makeSticky);
            ImGui::BeginDisabled(!_makeSticky);
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters().name("Max connections").max(parameters.cellMaxBonds).textWidth(MaxContentTextWidth), _maxConnections);
            ImGui::EndDisabled();
        }
        if (_mode == CreationMode::Drawing) {
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Ascending branch number").textWidth(MaxContentTextWidth), _ascendingBranchNumbers);
        }

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
    }
    ImGui::EndChild();
}

void _CreatorWindow::onDrawing()
{
    auto mousePos = ImGui::GetMousePos();
    auto pos = _viewport->mapViewToWorldPosition({mousePos.x, mousePos.y});
    if (!_drawing.isEmpty()) {
        _simController->removeSelectedEntities(false);
    }

    auto parameters = _simController->getSimulationParameters();
    auto maxConnections = !_makeSticky ? parameters.cellMaxBonds : _maxConnections;

    if (_drawing.isEmpty()) {
        auto cell = CellDescription()
                        .setId(NumberGenerator::getInstance().getId())
                        .setPos(pos)
                        .setEnergy(_energy)
                        .setMaxConnections(maxConnections)
                        .setTokenBranchNumber(_lastBranchNumber)
                        .setMetadata(CellMetadata().setColor(_editorModel->getDefaultColorCode()));
        _drawing.addCell(cell);
        incBranchNumber();
    } else {
        auto lastCellPos = _drawing.cells.back().pos;
        auto distance = Math::length(pos - lastCellPos);
        if (Math::length(pos - lastCellPos) >= _cellDistance) {
            for (float l = _cellDistance; l <= distance; l += _cellDistance) {
                auto cell = CellDescription()
                                .setId(NumberGenerator::getInstance().getId())
                                .setPos(lastCellPos + (pos - lastCellPos) * l / distance)
                                .setEnergy(_energy)
                                .setMaxConnections(maxConnections)
                                .setTokenBranchNumber(_lastBranchNumber)
                                .setMetadata(CellMetadata().setColor(_editorModel->getDefaultColorCode()));
                _drawing.addCell(cell);
                incBranchNumber();
            }
        }
    }
    DescriptionHelper::reconnectCells(_drawing, _cellDistance * 1.1f);
    if (!_makeSticky) {
        auto origDrawing = _drawing;
        DescriptionHelper::removeStickiness(_drawing);
        _simController->addAndSelectSimulationData(_drawing);
        _drawing = origDrawing;
    } else {
        _simController->addAndSelectSimulationData(_drawing);
    }

    _simController->reconnectSelectedEntities();
    _editorModel->update();
}

void _CreatorWindow::finishDrawing()
{
    _drawing.clear();
}

void _CreatorWindow::createCell()
{
    auto cell = CellDescription()
                    .setPos(getRandomPos())
                    .setEnergy(_energy)
                    .setMaxConnections(_maxConnections)
                    .setTokenBranchNumber(_lastBranchNumber)
                    .setMetadata(CellMetadata().setColor(_editorModel->getDefaultColorCode()));
    auto data = DataDescription().addCell(cell);
    _simController->addAndSelectSimulationData(data);
    incBranchNumber();
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
                                                  .removeStickiness(!_makeSticky)
                                                  .maxConnection(!_makeSticky ? parameters.cellMaxBonds : _maxConnections)
                                                  .color(_editorModel->getDefaultColorCode())
                                                  .center(getRandomPos()));

    _simController->addAndSelectSimulationData(data);
}

void _CreatorWindow::createHexagon()
{
    if (_layers <= 0) {
        return;
    }

    DataDescription data;
    auto parameters = _simController->getSimulationParameters();
    auto maxConnections = !_makeSticky ? parameters.cellMaxBonds : _maxConnections;

    auto incY = sqrt(3.0) * _cellDistance / 2.0;
    for (int j = 0; j < _layers; ++j) {
        for (int i = -(_layers - 1); i < _layers - j; ++i) {

            //create cell: upper layer
            data.addCell(CellDescription()
                             .setId(NumberGenerator::getInstance().getId())
                             .setEnergy(_energy)
                             .setPos({toFloat(i * _cellDistance + j * _cellDistance / 2.0), toFloat(-j * incY)})
                             .setMaxConnections(maxConnections)
                             .setMetadata(CellMetadata().setColor(_editorModel->getDefaultColorCode())));

            //create cell: under layer (except for 0-layer)
            if (j > 0) {
                data.addCell(CellDescription()
                                 .setId(NumberGenerator::getInstance().getId())
                                 .setEnergy(_energy)
                                 .setPos({toFloat(i * _cellDistance + j * _cellDistance / 2.0), toFloat(j * incY)})
                                 .setMaxConnections(maxConnections)
                                 .setMetadata(CellMetadata().setColor(_editorModel->getDefaultColorCode())));

            }
        }
    }

    DescriptionHelper::reconnectCells(data, _cellDistance * 1.5f);
    if (!_makeSticky) {
        DescriptionHelper::removeStickiness(data);
    }
    data.setCenter(getRandomPos());
    _simController->addAndSelectSimulationData(data);
}

void _CreatorWindow::createDisc()
{
    if (_innerRadius > _outerRadius || _innerRadius < 0 || _outerRadius <= 0) {
        return;
    }

    DataDescription data;
    auto parameters = _simController->getSimulationParameters();
    auto maxConnections = !_makeSticky ? parameters.cellMaxBonds : _maxConnections;
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
                             .setPos(relPos)
                             .setMaxConnections(maxConnections)
                             .setMetadata(CellMetadata().setColor(_editorModel->getDefaultColorCode())));
        }
    }

    DescriptionHelper::reconnectCells(data, _cellDistance * 1.7f);
    if (!_makeSticky) {
        DescriptionHelper::removeStickiness(data);
    }
    data.setCenter(getRandomPos());
    _simController->addAndSelectSimulationData(data);
}

void _CreatorWindow::drawing()
{
    if (!ImGui::GetIO().WantCaptureMouse) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            onDrawing();
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            finishDrawing();
        }
    }
}

RealVector2D _CreatorWindow::getRandomPos() const
{
    auto result = _viewport->getCenterInWorldPos();
    result.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    result.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    return result;
}

void _CreatorWindow::incBranchNumber()
{
    if (_ascendingBranchNumbers) {
        auto parameters = _simController->getSimulationParameters();
        _lastBranchNumber = (_lastBranchNumber + 1) % parameters.cellMaxTokenBranchNumber;
    }
}
