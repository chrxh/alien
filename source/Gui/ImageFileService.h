#pragma once

#include <filesystem>
#include <optional>

#include <Base/Singleton.h>

#include <EngineInterface/CreatorService.h>

class ImageFileService
{
    MAKE_SINGLETON(ImageFileService);

public:
    std::optional<RgbImage> loadRgbImage(std::filesystem::path const& path) const;
};
