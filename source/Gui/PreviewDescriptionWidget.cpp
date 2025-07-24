#include "PreviewDescriptionWidget.h"

#include "AlienGui.h"

PreviewDescriptionWidget _PreviewDescriptionWidget::create()
{
    return PreviewDescriptionWidget(new _PreviewDescriptionWidget());
}

bool _PreviewDescriptionWidget::process(PreviewDescription const& desc)
{
    return AlienGui::ShowPreviewDescription(desc, _zoom, _selectedNode);
}

float _PreviewDescriptionWidget::getZoom() const
{
    return _zoom;
}

void _PreviewDescriptionWidget::setZoom(float zoom)
{
    _zoom = zoom;
}

std::optional<int> _PreviewDescriptionWidget::getSelectedNode() const
{
    return _selectedNode;
}

void _PreviewDescriptionWidget::setSelectedNode(std::optional<int> selectedNode)
{
    _selectedNode = selectedNode;
}

_PreviewDescriptionWidget::_PreviewDescriptionWidget()
{
}