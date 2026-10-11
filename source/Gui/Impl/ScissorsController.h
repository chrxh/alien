#pragma once

#include <chrono>
#include <deque>

#include <Base/Singleton.h>

#include "Definitions.h"
#include "EditInteractionController.h"
#include "MainLoopEntity.h"

class ScissorsController
    : public MainLoopEntity
    , public EditInteractionController
{
    MAKE_SINGLETON(ScissorsController);

public:
    bool isCutOnlyInSelection() const;
    void setCutOnlyInSelection(bool value);

    void onLeftMouseButtonPressed(RealVector2D const& viewPos) override;
    void onLeftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos) override;

    void drawCursor(ImDrawList* drawList, ImVec2 const& mousePos) const override;

private:
    void init() override;
    void process() override;
    void shutdown() override;

    void cutConnections(RealVector2D const& startWorldPos, RealVector2D const& endWorldPos);

    bool _cutOnlyInSelection = false;

    struct TrailPoint
    {
        RealVector2D worldPos;
        std::chrono::steady_clock::time_point time;
        bool startsSegment = false;
    };
    std::deque<TrailPoint> _trail;
};
