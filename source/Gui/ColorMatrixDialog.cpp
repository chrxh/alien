#include "ColorMatrixDialog.h"

#include <algorithm>

#include <imgui.h>

#include <EngineInterface/NumberGenerator.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto const DialogSize = RealVector2D(560.0f, 480.0f);
    auto constexpr ButtonAreaHeight = 50.0f;

    float buttonWidth(std::string const& text)
    {
        return ImGui::CalcTextSize(text.c_str()).x + 2 * ImGui::GetStyle().FramePadding.x;
    }
}

ColorMatrixDialog::ColorMatrixDialog()
    : _modalWindow("", DialogSize)
{}

void ColorMatrixDialog::open(std::string const& name, unsigned int ownerId)
{
    _ownerId = ownerId;
    _modalWindow.setTitle(name);
    _modalWindow.open();
}

void ColorMatrixDialog::process(AlienGui::BasicInputColorMatrixParameters<bool> const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS], unsigned int ownerId)
{
    if (_ownerId != ownerId) {
        return;
    }

    _modalWindow.process([&] { processContent(parameters, value); });

    if (!_modalWindow.isOpen()) {
        _ownerId = 0;
    }
}

void ColorMatrixDialog::processContent(AlienGui::BasicInputColorMatrixParameters<bool> const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS])
{
    if (ImGui::BeginChild("##matrix", ImVec2(0, -scale(ButtonAreaHeight)), false)) {
        auto const& style = ImGui::GetStyle();
        auto available = ImGui::GetContentRegionAvail();
        auto numCells = toFloat(MAX_COLORS + 1);
        auto rowLabelWidth = ImGui::GetTextLineHeight() + style.ItemSpacing.x;
        auto cellSizeForWidth = (available.x - rowLabelWidth) / numCells - 2 * style.CellPadding.x - 1.0f;
        auto cellSizeForHeight = (available.y - ImGui::GetTextLineHeight() - style.ItemSpacing.y) / numCells - 2 * style.CellPadding.y;
        auto cellSize = std::max(std::min(cellSizeForWidth, cellSizeForHeight), 0.0f);
        auto blockWidth = rowLabelWidth + numCells * (cellSize + 2 * style.CellPadding.x + 1.0f);
        AlienGui::ColorMatrixBlock(parameters, value, std::min(available.x, blockWidth), cellSize);
    }
    ImGui::EndChild();

    AlienGui::Separator();

    if (AlienGui::Button("Close")) {
        _modalWindow.close();
        _ownerId = 0;
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    processToolButtons(parameters, value);
}

void ColorMatrixDialog::processToolButtons(AlienGui::BasicInputColorMatrixParameters<bool> const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS])
{
    auto forEachEditableEntry = [&](auto const& func) {
        for (int row = 0; row < MAX_COLORS; ++row) {
            for (int col = 0; col < MAX_COLORS; ++col) {
                if (parameters._disableDiagonal && row == col) {
                    continue;
                }
                func(value[row][col]);
            }
        }
    };

    auto toolButtonsWidth =
        buttonWidth("Clear") + buttonWidth("Select all") + buttonWidth("Invert") + buttonWidth("Randomize") + 4 * ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - toolButtonsWidth));

    if (AlienGui::Button("Clear")) {
        forEachEditableEntry([](bool& entry) { entry = false; });
    }
    ImGui::SameLine();
    if (AlienGui::Button("Select all")) {
        forEachEditableEntry([](bool& entry) { entry = true; });
    }
    ImGui::SameLine();
    if (AlienGui::Button("Invert")) {
        forEachEditableEntry([](bool& entry) { entry = !entry; });
    }
    ImGui::SameLine();
    if (AlienGui::Button("Randomize")) {
        forEachEditableEntry([](bool& entry) { entry = NumberGenerator::get().getRandomInt(2) == 0; });
    }
}
