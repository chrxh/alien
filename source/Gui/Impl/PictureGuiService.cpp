#include "PictureGuiService.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include <GLFW/glfw3.h>

#include <imgui.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <Base/AlienExceptions.h>
#include <Base/LoggingService.h>

#include <Rendering/Interface/RenderingFacade.h>

#include "PreviewDescView.h"
#include "SimulationView.h"
#include "StyleService.h"
#include "WindowController.h"

namespace
{
    auto constexpr JpgQuality = 92;

    auto constexpr PreviewPictureResolution = PictureGuiService::PreviewPictureResolution;
    auto constexpr PreviewPictureBrightness = 1.3f;
    auto constexpr PreviewPictureSupersampling = 2;

    // Relative to the font size, see OverlayController
    auto constexpr OverlayShadowOffset = 2.0f / 48.0f;

    // Leaves room for the smoothed edges of the overlay texts
    auto constexpr OverlayTextPadding = 2;
}

std::optional<std::string> PictureGuiService::createSimulationPreviewJpg()
{
    try {
        auto screenWidth = WindowController::get().getWindowData().mode->width;
        auto scaleFactor = toFloat(screenWidth) / toFloat(PreviewPictureResolution.x);
        auto renderResolution = IntVector2D{screenWidth, toInt(toFloat(PreviewPictureResolution.y) * scaleFactor)};

        auto picture = SimulationView::get().savePicture(renderResolution);
        auto preview = scale(picture, PreviewPictureResolution);
        return encodeJpg(brighten(preview, PreviewPictureBrightness));
    } catch (AlienException const& exception) {
        log(Priority::Important, std::string("preview picture could not be created: ") + exception.what());
        return std::nullopt;
    }
}

std::optional<std::string> PictureGuiService::createGenomePreviewJpg(std::vector<PreviewDesc> const& previews)
{
    if (previews.empty()) {
        return std::nullopt;
    }
    try {
        RealVector2D viewSize{toFloat(PreviewPictureResolution.x), toFloat(PreviewPictureResolution.y)};

        // The collage is built in an own draw list so that it can be rendered independently of the current ImGui frame
        ImDrawList drawList(ImGui::GetDrawListSharedData());
        drawList._ResetForNewFrame();
        drawList.PushTextureID(ImGui::GetIO().Fonts->TexID);
        drawList.PushClipRect({0, 0}, {viewSize.x, viewSize.y}, false);

        auto previewView = _PreviewDescView::create();
        previewView->drawCollage(&drawList, previews, {0, 0}, viewSize);

        drawList.PopClipRect();
        drawList.PopTextureID();

        auto picture =
            _RenderingFacade::get()->renderDrawList(&drawList, PreviewPictureResolution, Const::GenomePreviewBackgroundColor, PreviewPictureSupersampling);
        return encodeJpg(brighten(scale(picture, PreviewPictureResolution), PreviewPictureBrightness));
    } catch (std::exception const& exception) {
        log(Priority::Important, std::string("preview picture could not be created: ") + exception.what());
        return std::nullopt;
    }
}

PictureData PictureGuiService::scale(PictureData const& picture, IntVector2D const& resolution)
{
    if (resolution.x <= 0 || resolution.y <= 0) {
        throw AlienException("The resolution of a picture must be positive.");
    }

    PictureData result{.resolution = resolution, .pixels = std::vector<uint8_t>(static_cast<size_t>(resolution.x) * resolution.y * PictureData::NumChannels)};
    auto scaled = stbir_resize_uint8_srgb(
        picture.pixels.data(),
        picture.resolution.x,
        picture.resolution.y,
        picture.resolution.x * PictureData::NumChannels,
        result.pixels.data(),
        resolution.x,
        resolution.y,
        resolution.x * PictureData::NumChannels,
        STBIR_RGB);
    if (scaled == nullptr) {
        throw AlienException("The picture could not be scaled.");
    }
    return result;
}

PictureData PictureGuiService::brighten(PictureData const& picture, float factor)
{
    auto result = picture;
    for (auto& value : result.pixels) {
        value = static_cast<uint8_t>(std::min(255.0f, toFloat(value) * factor));
    }
    return result;
}

PictureGuiService::OverlayText PictureGuiService::renderOverlayText(std::string const& text, float fontSize)
{
    auto textSize = StyleService::get().getMonospaceLargeFont()->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, text.c_str());
    auto shadowOffset = fontSize * OverlayShadowOffset;
    IntVector2D resolution{
        toInt(std::ceil(textSize.x + shadowOffset)) + 2 * OverlayTextPadding, toInt(std::ceil(textSize.y + shadowOffset)) + 2 * OverlayTextPadding};

    auto render = [&](ImColor const& backgroundColor) {
        ImDrawList drawList(ImGui::GetDrawListSharedData());
        drawList._ResetForNewFrame();
        drawList.PushTextureID(ImGui::GetIO().Fonts->TexID);
        drawList.PushClipRect({0, 0}, {toFloat(resolution.x), toFloat(resolution.y)}, false);
        drawOverlayText(&drawList, text, fontSize, {toFloat(resolution.x - OverlayTextPadding), toFloat(resolution.y - OverlayTextPadding)});
        drawList.PopClipRect();
        drawList.PopTextureID();
        return _RenderingFacade::get()->renderDrawList(&drawList, resolution, backgroundColor, 1);
    };
    return OverlayText{.onBlack = render(ImColor(0, 0, 0)), .onWhite = render(ImColor(255, 255, 255))};
}

void PictureGuiService::drawOverlayText(PictureData& picture, OverlayText const& text, IntVector2D const& lowerRightCorner)
{
    // A pixel of the text lets the picture shine through as much as it appears brighter on white than on black
    auto const& size = text.onBlack.resolution;
    IntVector2D origin{lowerRightCorner.x + OverlayTextPadding - size.x, lowerRightCorner.y + OverlayTextPadding - size.y};
    for (int y = std::max(0, -origin.y); y < std::min(size.y, picture.resolution.y - origin.y); ++y) {
        for (int x = std::max(0, -origin.x); x < std::min(size.x, picture.resolution.x - origin.x); ++x) {
            auto textIndex = (static_cast<size_t>(y) * size.x + x) * PictureData::NumChannels;
            auto pictureIndex = (static_cast<size_t>(origin.y + y) * picture.resolution.x + origin.x + x) * PictureData::NumChannels;
            for (int channel = 0; channel < PictureData::NumChannels; ++channel) {
                auto onBlack = toInt(text.onBlack.pixels.at(textIndex + channel));
                auto onWhite = toInt(text.onWhite.pixels.at(textIndex + channel));
                auto& value = picture.pixels.at(pictureIndex + channel);
                value = static_cast<uint8_t>(std::min(255, onBlack + toInt(value) * std::max(0, onWhite - onBlack) / 255));
            }
        }
    }
}

void PictureGuiService::drawOverlayText(ImDrawList* drawList, std::string const& text, float fontSize, RealVector2D const& lowerRightCorner)
{
    // Like the overlay messages, the text casts a shadow to the upper left, see OverlayController
    auto font = StyleService::get().getMonospaceLargeFont();
    auto textSize = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, text.c_str());
    auto shadowOffset = fontSize * OverlayShadowOffset;
    ImVec2 position{lowerRightCorner.x - textSize.x, lowerRightCorner.y - textSize.y};
    drawList->AddText(font, fontSize, {position.x - shadowOffset, position.y - shadowOffset}, ImColor(0, 0, 0), text.c_str());
    drawList->AddText(font, fontSize, position, ImColor(255, 255, 255), text.c_str());
}

std::string PictureGuiService::encodeJpg(PictureData const& picture)
{
    std::string result;
    auto appendData = [](void* context, void* data, int size) { static_cast<std::string*>(context)->append(static_cast<char const*>(data), size); };
    auto writeResult =
        stbi_write_jpg_to_func(appendData, &result, picture.resolution.x, picture.resolution.y, PictureData::NumChannels, picture.pixels.data(), JpgQuality);
    if (writeResult == 0) {
        throw AlienException("The picture could not be encoded.");
    }
    return result;
}

std::string PictureGuiService::encodePng(PictureData const& picture)
{
    std::string result;
    auto appendData = [](void* context, void* data, int size) { static_cast<std::string*>(context)->append(static_cast<char const*>(data), size); };
    auto writeResult = stbi_write_png_to_func(
        appendData,
        &result,
        picture.resolution.x,
        picture.resolution.y,
        PictureData::NumChannels,
        picture.pixels.data(),
        picture.resolution.x * PictureData::NumChannels);
    if (writeResult == 0) {
        throw AlienException("The picture could not be encoded.");
    }
    return result;
}

void PictureGuiService::savePng(PictureData const& picture, std::filesystem::path const& filename)
{
    auto result = stbi_write_png(
        filename.string().c_str(),
        picture.resolution.x,
        picture.resolution.y,
        PictureData::NumChannels,
        picture.pixels.data(),
        picture.resolution.x * PictureData::NumChannels);
    if (result == 0) {
        throw AlienException("The file could not be written.");
    }
}
