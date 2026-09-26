#pragma once

#include <Base/Singleton.h>

#include "Definitions.h"
#include "EditInteractionController.h"
#include "MainLoopEntity.h"

class ForceController
    : public MainLoopEntity
    , public EditInteractionController
{
    MAKE_SINGLETON(ForceController);

public:
    void onLeftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos) override;

    void drawCursor(ImDrawList* drawList, ImVec2 const& mousePos) const override;

private:
    void init() override {}
    void process() override {}
    void shutdown() override {}
};
