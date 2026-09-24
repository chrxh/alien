#include "ImageFileService.h"

#include <stb_image.h>

std::optional<RgbImage> ImageFileService::loadRgbImage(std::filesystem::path const& path) const
{
    auto constexpr NumChannels = 3;

    int width, height, numChannels;
    auto pixels = stbi_load(path.string().c_str(), &width, &height, &numChannels, NumChannels);
    if (!pixels) {
        return std::nullopt;
    }
    auto result = RgbImage{.width = width, .height = height, .pixels = std::vector<uint8_t>(pixels, pixels + width * height * NumChannels)};
    stbi_image_free(pixels);
    return result;
}
