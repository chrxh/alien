#include "ResourcePreviewWidget.h"

#include <algorithm>

#include <glad/glad.h>

#include <imgui.h>

#include <Base/LoggingService.h>

#include "OpenGLHelper.h"
#include "PictureGuiService.h"
#include "StyleRepository.h"

void ResourcePreviewWidget::createForSimulation()
{
    clear();

    _jpg = PictureGuiService::get().createSimulationPreviewJpg().value_or(std::string());
    if (_jpg.empty()) {
        return;
    }
    try {
        _texture = OpenGLHelper::loadTextureFromMemory(_jpg);
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

void ResourcePreviewWidget::process()
{
    if (_texture.has_value()) {
        processSimulationPreview();
    }
    if (!_genomePreviews.empty()) {
        processGenomePreview();
    }
}

std::string const& ResourcePreviewWidget::getJpg() const
{
    return _jpg;
}

namespace
{
    // Reserving the scrollbar width independently of its visibility avoids a feedback loop between the picture height and the scrollbar
    float calcAvailableWidth()
    {
        auto const& style = ImGui::GetStyle();
        return ImGui::GetWindowWidth() - style.WindowPadding.x * 2 - style.ScrollbarSize;
    }
}

void ResourcePreviewWidget::processSimulationPreview()
{
    auto width = std::min(calcAvailableWidth(), scale(toFloat(_texture->width)));
    auto height = width * toFloat(_texture->height) / toFloat(_texture->width);
    ImGui::Image((ImTextureID)(intptr_t)_texture->textureId, {width, height});
}

void ResourcePreviewWidget::processGenomePreview()
{
    auto const& resolution = PictureGuiService::PreviewPictureResolution;

    // The aspect ratio of the uploaded picture makes the widget show the same tile layout
    auto width = calcAvailableWidth();
    auto height = width * toFloat(resolution.y) / toFloat(resolution.x);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, Const::GenomePreviewBackgroundColor.Value);
    if (ImGui::BeginChild("##genomePreview", ImVec2(width, height), 0, ImGuiWindowFlags_NoScrollbar)) {
        RealVector2D viewStartPos{ImGui::GetWindowPos().x, ImGui::GetWindowPos().y};
        RealVector2D viewSize{ImGui::GetWindowWidth(), ImGui::GetWindowHeight()};
        _renderer.drawCollage(ImGui::GetWindowDrawList(), _genomePreviews, viewStartPos, viewSize);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void ResourcePreviewWidget::clear()
{
    if (_texture.has_value()) {
        glDeleteTextures(1, &_texture->textureId);
        _texture.reset();
    }
    _jpg.clear();
    _genomePreviews.clear();
}
