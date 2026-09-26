#pragma once

#include <imgui.h>

#include <Base/Definitions.h>

// Handles the mouse input of an edit tool in the simulation view (positions in view coordinates)
class EditInteractionController
{
public:
    virtual ~EditInteractionController() = default;

    virtual void onLeftMouseButtonPressed(RealVector2D const& viewPos) {}
    virtual void onLeftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos) {}
    virtual void onLeftMouseButtonReleased(RealVector2D const& viewPos, RealVector2D const& prevViewPos) {}

    virtual void onRightMouseButtonPressed(RealVector2D const& viewPos) {}
    virtual void onRightMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos) {}
    virtual void onRightMouseButtonReleased() {}

    virtual bool isCrosshairCursor() const { return true; }
    virtual void drawCursor(ImDrawList* drawList, ImVec2 const& mousePos) const {}
};
