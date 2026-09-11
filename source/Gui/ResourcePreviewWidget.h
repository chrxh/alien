#pragma once

#include <optional>
#include <string>
#include <vector>

#include <EngineInterface/PreviewDesc.h>

#include <Network/Definitions.h>

#include "Definitions.h"
#include "PreviewDescView.h"

class ResourcePreviewWidget
{
public:
    void createForSimulation();
    void createForGenome(std::vector<PreviewDesc> const& previews);
    void process();

    std::string const& getJpg() const;

private:
    void processSimulationPreview();
    void processGenomePreview();
    void clear();

    std::string _jpg;
    std::optional<TextureData> _texture;

    std::vector<PreviewDesc> _genomePreviews;
    PreviewDescView _previewView = _PreviewDescView::create();
};
