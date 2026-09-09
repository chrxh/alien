#pragma once

#include <optional>
#include <string>

#include <Network/Definitions.h>

#include "Definitions.h"

class ResourcePreviewWidget
{
public:
    void create(NetworkResourceType resourceType);
    void process();

    std::optional<std::string> const& getJpg() const;

private:
    void createSimulationPreview();
    void clear();

    std::optional<std::string> _jpg;
    std::optional<TextureData> _texture;
};
