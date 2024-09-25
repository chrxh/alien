#include "Viewport.h"

#include <GLFW/glfw3.h>

#include "Base/Math.h"
#include "EngineInterface/SimulationController.h"

#include "WindowController.h"

SimulationController Viewport::_simController;
float Viewport::_zoomFactor = 1.0f;
float Viewport::_zoomSensitivity = 1.07f;
RealVector2D Viewport::_worldCenter;
IntVector2D Viewport::_viewSize;

void Viewport::init(SimulationController const& simController)
{
    _viewSize = WindowController::getStartupWindowSize();
    _simController = simController;
}

float Viewport::getZoomFactor()
{
    return _zoomFactor;
}

void Viewport::setZoomFactor(float zoomFactor)
{
    _zoomFactor = zoomFactor;
}

RealVector2D Viewport::getCenterInWorldPos()
{
    return _worldCenter;
}

void Viewport::setCenterInWorldPos(RealVector2D const& worldCenter)
{
    _worldCenter = worldCenter;
}

IntVector2D Viewport::getViewSize()
{
    return _viewSize;
}

void Viewport::setViewSize(IntVector2D const& viewSize)
{
    _viewSize = viewSize;
}

void Viewport::zoom(IntVector2D const& viewPos, float factor)
{
    if ((factor > 1.0f && _zoomFactor < 200.0f) || (factor < 1.0f && _zoomFactor > 0.02f)) {
        auto worldPos = mapViewToWorldPosition({toFloat(viewPos.x), toFloat(viewPos.y)});
        _zoomFactor *= factor;
        centerTo(worldPos, viewPos);
    }
}

float Viewport::getZoomSensitivity()
{
    return _zoomSensitivity;
}

void Viewport::setZoomSensitivity(float value)
{
    _zoomSensitivity = std::min(10.0f, std::max(1.0f, value));
}

void Viewport::centerTo(RealVector2D const& worldPosition, IntVector2D const& viewPos)
{
    RealVector2D deltaViewPos{
        toFloat(viewPos.x) - toFloat(_viewSize.x) / 2.0f, toFloat(viewPos.y) - toFloat(_viewSize.y) / 2.0f};
    auto deltaWorldPos = deltaViewPos / _zoomFactor;
    _worldCenter = worldPosition - deltaWorldPos;
}

RealVector2D Viewport::mapViewToWorldPosition(RealVector2D const& viewPos)
{
    RealVector2D relCenter{toFloat(_viewSize.x / (2.0 * _zoomFactor)), toFloat(_viewSize.y / (2.0 * _zoomFactor))};
    RealVector2D relWorldPos{viewPos.x / _zoomFactor, viewPos.y / _zoomFactor};
    return _worldCenter - relCenter + relWorldPos;
}

RealVector2D Viewport::mapWorldToViewPosition(RealVector2D worldPos, bool borderlessRendering)
{
    if (borderlessRendering) {
        auto worldSize = toRealVector2D(_simController->getWorldSize());
        auto offset = _worldCenter - worldSize / 2;
        worldPos.x = Math::modulo(worldPos.x - offset.x, worldSize.x) + offset.x;
        worldPos.y = Math::modulo(worldPos.y - offset.y, worldSize.y) + offset.y;
    }
    return {
        worldPos.x * _zoomFactor - _worldCenter.x * _zoomFactor + toFloat(_viewSize.x) / 2,
        worldPos.y * _zoomFactor - _worldCenter.y * _zoomFactor + toFloat(_viewSize.y) / 2};
}

RealRect Viewport::getVisibleWorldRect()
{
    auto topLeft = mapViewToWorldPosition(RealVector2D{0, 0});
    auto bottomRight = mapViewToWorldPosition(RealVector2D{toFloat(_viewSize.x - 1), toFloat(_viewSize.y - 1)});
    return {topLeft, bottomRight};
}

bool Viewport::isVisible(RealVector2D const& viewPos)
{
    return viewPos.x >= 0 && viewPos.y >= 0 && viewPos.x < _viewSize.x && viewPos.y < _viewSize.y;
}
