#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Data/PreviewDesc.h>

#include <Network/Definitions.h>

#include <Rendering/Interface/TextureData.h>

#include "Definitions.h"
#include "PreviewDescView.h"

class ResourcePreviewWidget
{
public:
    void createForSimulation();
    void createForPicture(std::string const& jpg);
    void createForGenome(std::vector<PreviewDesc> const& previews);
    void process();
    void process(float width, std::optional<std::string> const& label = std::nullopt);

    std::string const& getJpg() const;

private:
    void processSimulationPreview(float availableWidth, std::optional<std::string> const& label);
    void processGenomePreview(float width, std::optional<std::string> const& label);
    void clear();

    std::string _jpg;
    std::optional<TextureData> _texture;

    std::vector<PreviewDesc> _genomePreviews;
    PreviewDescView _previewView = _PreviewDescView::create();
};
