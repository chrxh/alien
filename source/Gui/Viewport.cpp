#include "Viewport.h"

float _Viewport::getZoomFactor() const
{
    return _zoomFactor;
}

void _Viewport::setZoomFactor(float zoomFactor)
{
    _zoomFactor = zoomFactor;
}

RealVector2D _Viewport::getCenterInWorldPos() const
{
    return _worldCenter;
}

void _Viewport::setCenterInWorldPos(RealVector2D const& worldCenter)
{
    _worldCenter = worldCenter;
}

IntVector2D _Viewport::getViewSize() const
{
    return _viewSize;
}

void _Viewport::setViewSize(IntVector2D const& viewSize)
{
    _viewSize = viewSize;
}

void _Viewport::zoom(IntVector2D const& viewPos, float factor)
{
    auto worldPos = mapViewToWorldPosition({toFloat(viewPos.x), toFloat(viewPos.y)});
    _zoomFactor *= factor;
    centerTo(worldPos, viewPos);
}

void _Viewport::centerTo(RealVector2D const& worldPosition, IntVector2D const& viewPos)
{
    RealVector2D deltaViewPos{
        toFloat(viewPos.x) - toFloat(_viewSize.x) / 2.0f, toFloat(viewPos.y) - toFloat(_viewSize.y) / 2.0f};
    auto deltaWorldPos = deltaViewPos / _zoomFactor;
    _worldCenter = worldPosition - deltaWorldPos;
}

RealVector2D _Viewport::mapViewToWorldPosition(RealVector2D const& viewPos) const
{
    RealVector2D relCenter{toFloat(_viewSize.x / (2.0 * _zoomFactor)), toFloat(_viewSize.y / (2.0 * _zoomFactor))};
    RealVector2D relWorldPos{viewPos.x / _zoomFactor, viewPos.y / _zoomFactor};
    return _worldCenter - relCenter + relWorldPos;
}

RealRect _Viewport::getVisibleWorldRect() const
{
    auto topLeft = mapViewToWorldPosition(RealVector2D{0, 0});
    auto bottomRight = mapViewToWorldPosition(RealVector2D{toFloat(_viewSize.x - 1), toFloat(_viewSize.y - 1)});
    return {topLeft, bottomRight};
}
