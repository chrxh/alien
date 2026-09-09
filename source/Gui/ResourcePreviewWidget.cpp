#include "ResourcePreviewWidget.h"

#include <algorithm>

#include <glad/glad.h>

#include <imgui.h>

#include <Base/LoggingService.h>

#include "OpenGLHelper.h"
#include "PictureGuiService.h"
#include "StyleRepository.h"

void ResourcePreviewWidget::create(NetworkResourceType resourceType)
{
    clear();

    // Genomes have no preview picture yet
    if (resourceType == NetworkResourceType_Simulation) {
        createSimulationPreview();
    }
}

void ResourcePreviewWidget::createSimulationPreview()
{
    _jpg = PictureGuiService::get().createSimulationPreviewJpg();
    if (!_jpg.has_value()) {
        return;
    }
    try {
        _texture = OpenGLHelper::loadTextureFromMemory(*_jpg);
    } catch (std::exception const&) {
        log(Priority::Important, "preview picture could not be decoded");
    }
}

void ResourcePreviewWidget::clear()
{
    if (_texture.has_value()) {
        glDeleteTextures(1, &_texture->textureId);
        _texture.reset();
    }
    _jpg.reset();
}

void ResourcePreviewWidget::process()
{
    if (!_texture.has_value()) {
        return;
    }

    // Reserving the scrollbar width independently of its visibility avoids a feedback loop between the picture height and the scrollbar
    auto const& style = ImGui::GetStyle();
    auto availableWidth = ImGui::GetWindowWidth() - style.WindowPadding.x * 2 - style.ScrollbarSize;
    auto width = std::min(availableWidth, scale(toFloat(_texture->width)));
    auto height = width * toFloat(_texture->height) / toFloat(_texture->width);
    ImGui::Image((ImTextureID)(intptr_t)_texture->textureId, {width, height});
}

std::optional<std::string> const& ResourcePreviewWidget::getJpg() const
{
    return _jpg;
}
