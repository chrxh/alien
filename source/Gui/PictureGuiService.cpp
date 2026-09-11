#include "PictureGuiService.h"

#include <algorithm>
#include <ranges>

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_opengl3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <Base/AlienExceptions.h>
#include <Base/ExitScopeGuard.h>
#include <Base/LoggingService.h>

#include "PreviewDescRenderer.h"
#include "SimulationView.h"
#include "StyleRepository.h"
#include "WindowController.h"

namespace
{
    auto constexpr JpgQuality = 70;

    auto constexpr PreviewPictureResolution = PictureGuiService::PreviewPictureResolution;
    auto constexpr PreviewPictureBrightness = 1.3f;
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

namespace
{
    // Renders a draw list into an offscreen framebuffer and reads the result back as a picture
    PictureData renderOffscreen(ImDrawList* drawList, IntVector2D const& resolution, ImColor const& backgroundColor)
    {
        GLint origFbo = 0;
        GLint origTexture = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &origFbo);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &origTexture);
        auto origScissorTest = glIsEnabled(GL_SCISSOR_TEST);

        GLuint texture = 0;
        GLuint fbo = 0;
        ExitScopeGuard restoreState([&] {
            glBindFramebuffer(GL_FRAMEBUFFER, origFbo);
            glDeleteFramebuffers(1, &fbo);
            glDeleteTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, origTexture);
            if (origScissorTest) {
                glEnable(GL_SCISSOR_TEST);
            }
        });

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution.x, resolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw AlienException("The offscreen framebuffer for the picture could not be created.");
        }

        ImDrawData drawData;
        drawData.Valid = true;
        drawData.DisplayPos = {0, 0};
        drawData.DisplaySize = {toFloat(resolution.x), toFloat(resolution.y)};
        drawData.FramebufferScale = {1.0f, 1.0f};
        drawData.AddDrawList(drawList);

        // A scissor rect left over from the surrounding frame would clip the clear
        glDisable(GL_SCISSOR_TEST);
        glClearColor(backgroundColor.Value.x, backgroundColor.Value.y, backgroundColor.Value.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(&drawData);

        PictureData result{
            .resolution = resolution, .pixels = std::vector<uint8_t>(static_cast<size_t>(resolution.x) * resolution.y * PictureData::NumChannels)};
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, resolution.x, resolution.y, GL_RGB, GL_UNSIGNED_BYTE, result.pixels.data());

        // OpenGL provides the rows bottom-up
        auto bytesPerRow = static_cast<size_t>(resolution.x) * PictureData::NumChannels;
        for (auto row : std::views::iota(0, resolution.y / 2)) {
            auto upperRow = result.pixels.begin() + row * bytesPerRow;
            auto lowerRow = result.pixels.begin() + (resolution.y - 1 - row) * bytesPerRow;
            std::swap_ranges(upperRow, upperRow + bytesPerRow, lowerRow);
        }
        return result;
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

        PreviewDescRenderer renderer;
        renderer.drawCollage(&drawList, previews, {0, 0}, viewSize);

        drawList.PopClipRect();
        drawList.PopTextureID();

        return encodeJpg(renderOffscreen(&drawList, PreviewPictureResolution, Const::GenomePreviewBackgroundColor));
    } catch (AlienException const& exception) {
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
