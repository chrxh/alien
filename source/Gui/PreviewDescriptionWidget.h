#pragma once

#include "EngineInterface/PreviewDescriptions.h"

#include "Definitions.h"

class _PreviewDescriptionWidget
{
public:
    static PreviewDescriptionWidget create();

    bool process(PreviewDescription const& desc);
    
    float getZoom() const;
    void setZoom(float zoom);
    
    std::optional<int> getSelectedNode() const;
    void setSelectedNode(std::optional<int> selectedNode);

private:
    _PreviewDescriptionWidget();

    float _zoom = 20.0f;
    std::optional<int> _selectedNode;
};