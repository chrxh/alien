#include "CreaturePreviewWidget.h"

#include <algorithm>
#include <cmath>
#include <ranges>

#include <boost/algorithm/string/join.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/PreviewDescConverterService.h>
#include <EngineInterface/SpaceCalculator.h>

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "GenomeWindowEditData.h"
#include "SimulationScrollbars.h"
#include "StyleService.h"

namespace
{
    auto constexpr NeuralActivityTextMargin = 15.0f;
    auto constexpr NeuralActivitySliderWidth = 55.0f;
    auto constexpr NeuralActivitySlidersPerColumn = 4;
    auto constexpr NeuralActivityEditorRightMargin = 30.0f;
    auto constexpr TitleMargin = 7.0f;

    std::string getNeuralActivityEditorSignalLabel(int index)
    {
        return "Signal " + std::to_string(index);
    }

    std::string getNeuralActivityEditorMemoryLabel(int index)
    {
        return "Mem " + std::to_string(index);
    }

    // The labels are placed right of the sliders and would be clipped by the column boundary if the text column were too narrow
    float calcNeuralActivityTextWidth()
    {
        auto outLabel = getNeuralActivityEditorSignalLabel(STANDARD_NEURONS_PER_CELL - 1);
        auto memoryLabel = getNeuralActivityEditorMemoryLabel(MEMORY_NEURONS_PER_CELL - 1);
        auto labelWidth = std::max(ImGui::CalcTextSize(outLabel.c_str()).x, ImGui::CalcTextSize(memoryLabel.c_str()).x);
        return scaleInverse(labelWidth) + NeuralActivityTextMargin;
    }

    float calcNeuralActivityColumnWidth()
    {
        return NeuralActivitySliderWidth + calcNeuralActivityTextWidth();
    }

    // The neural activity editor exposes the outgoing signal channels and the memory activities of the selected cell
    int calcNumNeuralActivityEditorColumns()
    {
        return (STANDARD_NEURONS_PER_CELL + MEMORY_NEURONS_PER_CELL) / NeuralActivitySlidersPerColumn;
    }
}


CreaturePreviewWidget
_CreaturePreviewWidget::create(GenomeTabEditData const& editData, GeneIndicesForSubGenome const& geneIndices, SubGenomeDesc const& genomeWithStartIndex)
{
    return CreaturePreviewWidget(new _CreaturePreviewWidget(editData, geneIndices, genomeWithStartIndex));
}

void _CreaturePreviewWidget::process(bool& phenotypeChanged, ContentDesc& phenotype, GenomeDesc const& genome, float height)
{
    auto phenotypeWithoutSeed = phenotype;
    GenomeDescEditService::get().removeSeedFromPhenotype(phenotypeWithoutSeed);

    auto geneStartIndex = _subGenome.startIndex;

    auto conversionResult = PreviewDescConverterService::get().convertToPreviewDesc(genome, geneStartIndex, std::move(phenotypeWithoutSeed), _visualFrontAngle);
    _visualFrontAngle = conversionResult.visualFrontAngle;
    _previewDesc = std::move(conversionResult.description);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, Const::GenomePreviewBackgroundColor.Value);

    if (ImGui::BeginChild("CellGraphWidget", ImVec2(0, height), 0, ImGuiWindowFlags_NoScrollbar)) {
        updateViewport();
        processMouseNavigation();
        updateSelection();
        processPreviewRendering();
        processTitle();
        processNeuralActivityEditor(phenotypeChanged, phenotype);
        processActionButtons();
        processScrollbars();
    }
    ImGui::EndChild();

    ImGui::PopStyleColor();
}

uint64_t _CreaturePreviewWidget::getCreatureId() const
{
    return _creatureId;
}

void _CreaturePreviewWidget::setCreatureId(uint64_t value)
{
    _creatureId = value;
}

GeneIndicesForSubGenome const& _CreaturePreviewWidget::getGeneIndices() const
{
    return _geneIndices;
}

void _CreaturePreviewWidget::setGeneIndices(GeneIndicesForSubGenome const& value)
{
    _geneIndices = value;
}

SubGenomeDesc const& _CreaturePreviewWidget::getGenomeWithStartIndex() const
{
    return _subGenome;
}

void _CreaturePreviewWidget::setGenomeWithStartIndex(SubGenomeDesc const& value)
{
    _subGenome = value;
}

PreviewDesc const& _CreaturePreviewWidget::getPreviewDesc() const
{
    return _previewDesc;
}

void _CreaturePreviewWidget::resetVisualFrontAngle()
{
    _visualFrontAngle.reset();
}

_CreaturePreviewWidget::_CreaturePreviewWidget(
    GenomeTabEditData const& editData,
    GeneIndicesForSubGenome const& geneIndices,
    SubGenomeDesc const& genomeWithStartIndex)
    : _editData(editData)
    , _geneIndices(geneIndices)
    , _subGenome(genomeWithStartIndex)
{
    _scrollbars = std::make_shared<_SimulationScrollbars>(false);
    _previewView = _PreviewDescView::create();
}

void _CreaturePreviewWidget::updateViewport()
{
    _viewport.setViewStartPos({ImGui::GetWindowPos().x, ImGui::GetWindowPos().y});
    _viewport.setViewSize({ImGui::GetWindowWidth(), ImGui::GetWindowHeight()});
}

void _CreaturePreviewWidget::processMouseNavigation()
{
    // The mouse wheel is reserved for zooming, otherwise ImGui would forward it to the scrollable parent window
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
        ImGui::SetKeyOwner(ImGuiKey_MouseWheelY, ImGui::GetCurrentWindow()->ID);
    }

    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
        RealVector2D mousePos = {ImGui::GetMousePos().x, ImGui::GetMousePos().y};

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            _worldPosForPanning = _viewport.mapViewToWorldPosition(mousePos);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) && _worldPosForPanning.has_value()) {
            moveCenter(_worldPosForPanning.value(), mousePos);
        }
        if (ImGui::GetIO().MouseWheel > 0) {
            auto worldPos = _viewport.mapViewToWorldPosition(mousePos);
            _viewport.setZoom(_viewport.getZoom() * sqrt(1.5f));
            moveCenter(worldPos, mousePos);
        }
        if (ImGui::GetIO().MouseWheel < 0) {
            auto worldPos = _viewport.mapViewToWorldPosition(mousePos);
            _viewport.setZoom(_viewport.getZoom() / sqrt(1.5f));
            moveCenter(worldPos, mousePos);
        }
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
        _worldPosForPanning.reset();
    }
}

void _CreaturePreviewWidget::processPreviewRendering()
{
    auto parameters = _PreviewDescView::RenderParameters()
                          .showFrontMarker(true)
                          .showGeneReferences(true)
                          .cellLabel(_editData->showNodeIndex ? _PreviewDescView::CellLabel::NodeIndex : _PreviewDescView::CellLabel::CellType)
                          .selectedGeneIndex(_editData->selectedGeneIndex)
                          .selectedNodeIndex(_editData->getSelectedNodeIndex())
                          .selectedCellId(_selectedCellIdFromPreview);
    _previewView->draw(ImGui::GetWindowDrawList(), _previewDesc, _viewport, parameters);
}

void _CreaturePreviewWidget::updateSelection()
{
    auto const cellSize = scale(_viewport.getZoom());
    auto mousePos = ImGui::GetMousePos();
    auto clickedOnPreviewWindow = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

    // Clear selection if another node has been selected outside of this widget or if cell id does not exist in preview
    auto selectedCellIdExists = false;
    for (auto const& cell : _previewDesc._cells) {
        if (_selectedCellIdFromPreview.has_value() && _selectedCellIdFromPreview.value() == cell._id) {
            selectedCellIdExists = true;
            break;
        }
    }
    if (!selectedCellIdExists || _selectedNodeFromPreview != _editData->getSelectedNodeIndex()) {
        _selectedCellIdFromPreview.reset();
        _selectedNodeFromPreview.reset();
    }

    if (!clickedOnPreviewWindow) {
        return;
    }
    for (auto const& cell : _previewDesc._cells) {
        auto cellPos = _viewport.mapWorldToViewPosition(cell._pos);
        if (mousePos.x >= cellPos.x - cellSize / 2 && mousePos.y >= cellPos.y - cellSize / 2 && mousePos.x <= cellPos.x + cellSize / 2
            && mousePos.y <= cellPos.y + cellSize / 2) {
            if (_editData->hasValidNodeIndex(cell._geneIndex, cell._nodeIndex)) {
                _selectedNodeFromPreview = cell._nodeIndex;
                _selectedCellIdFromPreview = cell._id;

                _editData->selectedGeneIndex = cell._geneIndex;
                _editData->setSelectedNodeIndex(cell._nodeIndex);
            } else {
                _selectedNodeFromPreview.reset();
                _selectedCellIdFromPreview.reset();
            }
        }
    }
}

namespace
{
    float calcCardHeaderTextOffsetY()
    {
        auto const& style = ImGui::GetStyle();
        return style.FramePadding.y + style.ItemSpacing.y;
    }

}

bool _CreaturePreviewWidget::processHideButton()
{
    auto const& style = ImGui::GetStyle();
    auto buttonWidth = ImGui::CalcTextSize(ICON_FA_TIMES).x + style.FramePadding.x * 2;
    auto contentStartPos = ImGui::GetCursorStartPos();

    auto cursorPos = ImGui::GetCursorPos();
    ImGui::SetCursorPos({ImGui::GetWindowWidth() - contentStartPos.x - buttonWidth, contentStartPos.y + style.ItemSpacing.y - style.FramePadding.y});
    auto result = AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_TIMES).tooltip("Hide the neural activity editor"));
    ImGui::SetCursorPos(cursorPos);

    return result;
}

void _CreaturePreviewWidget::processNeuralActivityEditor(bool& phenotypeChanged, ContentDesc& phenotype)
{
    auto contentAvailable = ImGui::GetContentRegionAvail();
    if (contentAvailable.x < scale(480.0f) || contentAvailable.y < scale(250.0f)) {
        return;
    }
    if (!_editData->showNeuralActivityEditor) {
        return;
    }

    auto editorWidth = calcNeuralActivityColumnWidth() * toFloat(calcNumNeuralActivityEditorColumns()) + 30.0f;
    auto width = _editData->detailSimulation && _selectedCellIdFromPreview.has_value() ? scale(editorWidth) : scale(250);
    auto height = _editData->detailSimulation && _selectedCellIdFromPreview.has_value() ? scale(149.0f) : scale(67.0f);
    ImGui::SetCursorPos(
        {ImGui::GetScrollX() + ImGui::GetWindowWidth() - width - scale(NeuralActivityEditorRightMargin), ImGui::GetScrollY() + scale(TitleMargin)});

    // The frame style derives the background of the editor from the frame color, therefore it is overridden only for the child window itself
    ImGui::PushStyleColor(ImGuiCol_FrameBg, static_cast<ImVec4>(Const::FloatingCardBackgroundColor));
    auto signalEditorVisible = ImGui::BeginChild("signalEditor", ImVec2(width, height), ImGuiChildFlags_FrameStyle);
    ImGui::PopStyleColor();
    if (signalEditorVisible) {

        AlienGui::Group(AlienGui::GroupParameters().text("Neural activity editor"));
        if (processHideButton()) {
            _editData->showNeuralActivityEditor = false;
        }

        if (_editData->detailSimulation && _selectedCellIdFromPreview.has_value()) {
            std::optional<CellPreviewDesc> selectedCell;
            for (auto const& cell : _previewDesc._cells) {
                if (cell._id == _selectedCellIdFromPreview.value()) {
                    selectedCell = cell;
                    break;
                }
            }
            CHECK(selectedCell.has_value());

            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));  // Transparent background
            ImGuiStyle& style = ImGui::GetStyle();
            auto originalGrabMinSize = style.GrabMinSize;
            style.GrabMinSize = scale(8.0f);

            struct NeuralActivityEditorEntry
            {
                std::string name;
                float value;
            };
            std::vector<NeuralActivityEditorEntry> entries;
            for (auto index : std::views::iota(0, STANDARD_NEURONS_PER_CELL)) {
                entries.emplace_back(getNeuralActivityEditorSignalLabel(index), selectedCell->_signal._channels.at(index));
            }
            for (auto index : std::views::iota(0, MEMORY_NEURONS_PER_CELL)) {
                entries.emplace_back(getNeuralActivityEditorMemoryLabel(index), selectedCell->_memory.at(index));
            }

            auto textWidth = calcNeuralActivityTextWidth();
            auto numColumns = calcNumNeuralActivityEditorColumns();
            for (auto column : std::views::iota(0, numColumns)) {
                ImGui::PushID(column);
                if (ImGui::BeginChild("", ImVec2(scale(calcNeuralActivityColumnWidth() - 5.0f), scale(0)))) {
                    for (auto row : std::views::iota(0, NeuralActivitySlidersPerColumn)) {
                        auto& entry = entries.at(column * NeuralActivitySlidersPerColumn + row);
                        phenotypeChanged |= AlienGui::SliderFloat(
                            AlienGui::SliderFloatParameters().name(entry.name).format("%.3f").textWidth(textWidth).min(-2.0f).max(2.0f), &entry.value);
                    }
                }
                ImGui::EndChild();
                ImGui::PopID();
                if (column < numColumns - 1) {
                    ImGui::SameLine();
                }
            }

            for (auto [channel, entry] : std::views::zip(selectedCell->_signal._channels, entries)) {
                channel = entry.value;
            }
            for (auto [memory, entry] : std::views::zip(selectedCell->_memory, entries | std::views::drop(STANDARD_NEURONS_PER_CELL))) {
                memory = entry.value;
            }

            style.GrabMinSize = originalGrabMinSize;
            ImGui::PopStyleColor();

            if (phenotypeChanged) {
                updatePhenotype(phenotype, selectedCell.value());
            }
        } else {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + scale(AlienGui::GroupTextIndent));
            if (!_editData->detailSimulation) {
                AlienGui::Text("Detailed simulation mode disabled");
            } else if (!_selectedCellIdFromPreview.has_value()) {
                AlienGui::Text("No cell selected");
            }
        }
    }
    ImGui::EndChild();
}

void _CreaturePreviewWidget::processActionButtons()
{
    ImGui::SetCursorPos({ImGui::GetScrollX() + scale(10.0f), ImGui::GetScrollY() + ImGui::GetWindowHeight() - scale(40.0f)});
    if (ImGui::BeginChild("##buttons", ImVec2(scale(110), scale(30)), 0)) {
        ImGui::SetCursorPos({0, 0});
        ImGui::PushID(1);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_SEARCH_PLUS))) {
            _viewport.setZoom(_viewport.getZoom() * 1.5f);
        }
        ImGui::PopID();
        ImGui::SameLine();
        ImGui::PushID(2);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_SEARCH_MINUS))) {
            _viewport.setZoom(_viewport.getZoom() / 1.5f);
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
}

void _CreaturePreviewWidget::processScrollbars()
{
    auto viewStartPos = _viewport.getViewStartPos();
    auto viewSize = _viewport.getViewSize();

    RealRect worldRect{{-100.0f, -100.0f}, {100.0f, 100.0f}};
    RealRect visibleWorldRect{
        _viewport.mapViewToWorldPosition(viewStartPos),
        _viewport.mapViewToWorldPosition(viewStartPos + viewSize),
    };
    RealRect viewRect{viewStartPos, viewStartPos + viewSize};

    auto worldCenter = _viewport.getWorldCenter();
    _scrollbars->process(worldCenter, worldRect, visibleWorldRect, viewRect);
    _viewport.setWorldCenter(worldCenter);
}

void _CreaturePreviewWidget::processTitle()
{
    ImGui::SetCursorPos({scale(TitleMargin), scale(TitleMargin) + calcCardHeaderTextOffsetY()});
    std::vector<std::string> geneIndexStrings;
    auto geneIndices = getGeneIndices();
    for (auto const& geneIndex : geneIndices) {
        geneIndexStrings.emplace_back(std::to_string(geneIndex));
    }
    auto subGenomeType = _subGenome.startIndex == 0 ? "Primary" : "Secondary";
    auto numCells = std::ranges::count_if(_previewDesc._cells, [](auto const& cell) { return cell._cellType != CellType_Void; });
    auto cellCountText = std::to_string(numCells) + " cells" + (_subGenome.trimmed ? " (trimmed)" : "");
    auto title = std::string(subGenomeType) + ": " + cellCountText + ", gene indices: " + boost::join(geneIndexStrings, ", ");
    AlienGui::Text(title.c_str());
}

void _CreaturePreviewWidget::moveCenter(RealVector2D const& startWorldPosition, RealVector2D const& endViewPos)
{
    auto deltaViewPos = endViewPos - _viewport.getViewStartPos() - _viewport.getViewSize() / 2.0f;
    auto deltaWorldPos = deltaViewPos / _viewport.getZoom();
    _viewport.setWorldCenter(startWorldPosition - deltaWorldPos);
}

void _CreaturePreviewWidget::updatePhenotype(ContentDesc& phenotype, CellPreviewDesc const& editedCell) const
{
    for (auto& object : phenotype._objects) {
        if (object._id == editedCell._id) {
            object.getCellRef()._neuralActivity = NeuralActivityDesc().signals(editedCell._signal._channels).memory(editedCell._memory);
        }
    }
}
