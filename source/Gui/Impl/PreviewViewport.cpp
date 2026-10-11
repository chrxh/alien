#include "PreviewViewport.h"

#include <algorithm>

#include "StyleService.h"

float PreviewViewport::getZoom() const
{
    return _zoom;
}

void PreviewViewport::setZoom(float value)
{
    _zoom = value;
}

RealVector2D PreviewViewport::getWorldCenter() const
{
    return _worldCenter;
}

void PreviewViewport::setWorldCenter(RealVector2D const& value)
{
    _worldCenter = value;
}

RealVector2D PreviewViewport::getViewStartPos() const
{
    return _viewStartPos;
}

void PreviewViewport::setViewStartPos(RealVector2D const& value)
{
    _viewStartPos = value;
}

RealVector2D PreviewViewport::getViewSize() const
{
    return _viewSize;
}

void PreviewViewport::setViewSize(RealVector2D const& value)
{
    _viewSize = value;
}

RealVector2D PreviewViewport::mapWorldToViewPosition(RealVector2D const& worldPos) const
{
    auto scaleFactor = scale(_zoom);
    return {
        (worldPos.x - _worldCenter.x) * scaleFactor + _viewSize.x / 2 + _viewStartPos.x,
        (worldPos.y - _worldCenter.y) * scaleFactor + _viewSize.y / 2 + _viewStartPos.y};
}

RealVector2D PreviewViewport::mapViewToWorldPosition(RealVector2D const& viewPos) const
{
    auto scaleFactor = scale(_zoom);
    return {
        (viewPos.x - _viewStartPos.x - _viewSize.x / 2) / scaleFactor + _worldCenter.x,
        (viewPos.y - _viewStartPos.y - _viewSize.y / 2) / scaleFactor + _worldCenter.y};
}

float PreviewViewport::calcZoomToFitContent(RealVector2D const& contentSize, RealVector2D const& viewSize)
{
    return scaleInverse(std::min(viewSize.x / contentSize.x, viewSize.y / contentSize.y));
}
