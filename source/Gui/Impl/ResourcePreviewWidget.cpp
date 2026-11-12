#include "ResourcePreviewWidget.h"

#include <algorithm>

#include <imgui.h>

#include <Base/LoggingService.h>

#include <Rendering/Interface/RenderingFacade.h>

#include "PictureGuiService.h"
#include "StyleService.h"

void ResourcePreviewWidget::createForSimulation()
{
    clear();

    _jpg = PictureGuiService::get().createSimulationPreviewJpg().value_or(std::string());
    if (_jpg.empty()) {
        return;
    }
    try {
        _texture = _RenderingFacade::get()->loadTextureFromMemory(_jpg);
    } catch (std::exception const&) {
        log(Priority::Important, "preview picture could not be decoded");
    }
}

void ResourcePreviewWidget::createForGenome(std::vector<PreviewDesc> const& previews)
{
    clear();

    for (auto const& preview : previews) {
        if (!preview._cells.empty()) {
            _genomePreviews.emplace_back(preview);
        }
    }

    // The uploaded picture shows the same collage as the widget
    _jpg = PictureGuiService::get().createGenomePreviewJpg(_genomePreviews).value_or(std::string());
}

namespace
{
    auto constexpr LabelHeight = 24.0f;
    auto constexpr LabelMargin = 10.0f;
    auto constexpr LabelPadding = 10.0f;

    // Reserving the scrollbar width independently of its visibility avoids a feedback loop between the picture height and the scrollbar
    float calcAvailableWidth()
    {
        auto const& style = ImGui::GetStyle();
        return ImGui::GetWindowWidth() - style.WindowPadding.x * 2 - style.ScrollbarSize;
    }

    void drawLabel(ImDrawList* drawList, ImVec2 const& pictureMin, ImVec2 const& pictureMax, std::optional<std::string> const& label)
    {
        if (!label.has_value()) {
            return;
        }
        auto size = ImVec2{ImGui::CalcTextSize(label->c_str()).x + scale(LabelPadding) * 2, scale(LabelHeight)};
        auto pos = ImVec2{pictureMin.x + scale(LabelMargin), pictureMax.y - scale(LabelMargin) - size.y};
        drawList->AddRectFilled(pos, {pos.x + size.x, pos.y + size.y}, ImColor::HSV(0.62f, 0.6f, 0.05f, 0.62f), size.y / 2);
        drawList->AddText({pos.x + scale(LabelPadding), pos.y + (size.y - ImGui::GetTextLineHeight()) / 2}, ImGui::GetColorU32(ImGuiCol_Text), label->c_str());
    }
}

void ResourcePreviewWidget::process()
{
    process(calcAvailableWidth());
}

void ResourcePreviewWidget::process(float width, std::optional<std::string> const& label)
{
    if (_texture.has_value()) {
        processSimulationPreview(width, label);
    }
    if (!_genomePreviews.empty()) {
        processGenomePreview(width, label);
    }
}

std::string const& ResourcePreviewWidget::getJpg() const
{
    return _jpg;
}

void ResourcePreviewWidget::processSimulationPreview(float availableWidth, std::optional<std::string> const& label)
{
    auto width = std::min(availableWidth, scale(toFloat(_texture->width)));
    auto height = width * toFloat(_texture->height) / toFloat(_texture->width);
    ImGui::Image(_texture->textureId, {width, height});
    drawLabel(ImGui::GetWindowDrawList(), ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), label);
}

void ResourcePreviewWidget::processGenomePreview(float width, std::optional<std::string> const& label)
{
    auto const& resolution = PictureGuiService::PreviewPictureResolution;

    // The aspect ratio of the uploaded picture makes the widget show the same tile layout
    auto height = width * toFloat(resolution.y) / toFloat(resolution.x);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, Const::GenomePreviewBackgroundColor.Value);
    if (ImGui::BeginChild("##genomePreview", ImVec2(width, height), 0, ImGuiWindowFlags_NoScrollbar)) {
        auto windowPos = ImGui::GetWindowPos();
        auto windowSize = ImGui::GetWindowSize();
        RealVector2D viewStartPos{windowPos.x, windowPos.y};
        RealVector2D viewSize{windowSize.x, windowSize.y};
        _previewView->drawCollage(ImGui::GetWindowDrawList(), _genomePreviews, viewStartPos, viewSize);
        drawLabel(ImGui::GetWindowDrawList(), windowPos, {windowPos.x + windowSize.x, windowPos.y + windowSize.y}, label);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void ResourcePreviewWidget::clear()
{
    if (_texture.has_value()) {
        _RenderingFacade::get()->deleteTexture(*_texture);
        _texture.reset();
    }
    _jpg.clear();
    _genomePreviews.clear();
}
