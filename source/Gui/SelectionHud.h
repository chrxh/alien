#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/SelectionShallowData.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class SelectionHud : public MainLoopEntity
{
    MAKE_SINGLETON(SelectionHud);

private:
    void init() override {}
    void process() override;
    void shutdown() override {}

    struct ViewBounds
    {
        RealVector2D center;
        RealVector2D topLeft;
        RealVector2D bottomRight;
    };
    ViewBounds calcViewBounds() const;

    void processFrame(ViewBounds const& bounds) const;
    void processRotationHandle(ViewBounds const& bounds);
    void processVelocityHandle(ViewBounds const& bounds);
    void processSummary(ViewBounds const& bounds) const;
    void processActionBar(ViewBounds const& bounds);

    void processColorPopup();
    void processStickyPopup();
    void processFixedPopup();
    void processMultiplyPopup();
    void processInspectPopup();
    void processMorePopup();

    std::optional<float> _lastRotationAngle;
    std::optional<RealVector2D> _lastVelocityHandlePos;

    float _angle = 0;
    float _angularVelocity = 0;
    std::optional<SelectionShallowData> _lastSelection;
};
