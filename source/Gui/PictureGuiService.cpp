#include "PictureGuiService.h"

#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <Base/AlienExceptions.h>

namespace
{
    auto constexpr JpgQuality = 70;
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
