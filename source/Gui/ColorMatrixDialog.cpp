#include "ColorMatrixDialog.h"

#include <algorithm>
#include <ranges>

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

    template <typename T>
    void processToolButtons(bool, ColorMatrix<T>&)
    {}

    void processToolButtons(bool disableDiagonal, ColorMatrix<bool>& matrix)
    {
        auto forEachEditableEntry = [&](auto const& func) {
            for (int row = 0; row < MAX_COLORS; ++row) {
                for (int col = 0; col < MAX_COLORS; ++col) {
                    if (disableDiagonal && row == col) {
                        continue;
                    }
                    func(matrix[row][col]);
                }
            }
        };

        ImGui::SameLine();
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
}

template <typename T>
ColorMatrixDialog<T>::ColorMatrixDialog()
    : _modalWindow("", DialogSize)
{}

template <typename T>
void ColorMatrixDialog<T>::open(
    AlienGui::BasicInputColorMatrixParameters<T> const& parameters,
    T const (&value)[MAX_COLORS][MAX_COLORS],
    std::function<void(ColorMatrix<T> const&)> const& onAdoptCallback)
{
    _parameters = parameters;
    for (auto const& [matrixRow, valueRow] : std::views::zip(_matrix.values, value)) {
        std::ranges::copy(valueRow, std::begin(matrixRow));
    }
    _onAdoptCallback = onAdoptCallback;
    _modalWindow.setTitle(parameters._name);
    _modalWindow.open();
}

template <typename T>
void ColorMatrixDialog<T>::process()
{
    _modalWindow.process([this] { processContent(); });
}

template <typename T>
void ColorMatrixDialog<T>::processContent()
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
        AlienGui::ExpandedColorMatrix(_parameters, _matrix.values, std::min(available.x, blockWidth), cellSize);
    }
    ImGui::EndChild();

    AlienGui::Separator();

    if (AlienGui::Button("Adopt")) {
        onAdopt();
        _modalWindow.close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        _modalWindow.close();
    }

    processToolButtons(_parameters._disableDiagonal, _matrix);
}

template <typename T>
void ColorMatrixDialog<T>::onAdopt()
{
    if (_onAdoptCallback) {
        _onAdoptCallback(_matrix);
    }
}

template class ColorMatrixDialog<bool>;
template class ColorMatrixDialog<int>;
template class ColorMatrixDialog<float>;
