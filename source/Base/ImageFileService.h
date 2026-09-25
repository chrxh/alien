#pragma once

#include <filesystem>
#include <optional>

#include "RgbImage.h"
#include "Singleton.h"

class ImageFileService
{
    MAKE_SINGLETON(ImageFileService);

public:
    std::optional<RgbImage> loadRgbImage(std::filesystem::path const& path) const;
};
