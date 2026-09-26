#pragma once

#include <Base/Singleton.h>

#include "Definitions.h"
#include "EditInteractionController.h"
#include "MainLoopEntity.h"

class SelectionController
    : public MainLoopEntity
    , public EditInteractionController
{
    MAKE_SINGLETON(SelectionController);

public:
    void onLeftMouseButtonPressed(RealVector2D const& viewPos) override;
    void onLeftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos) override;
    void onLeftMouseButtonReleased(RealVector2D const& viewPos, RealVector2D const& prevViewPos) override;

    void onRightMouseButtonPressed(RealVector2D const& viewPos) override;
    void onRightMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos) override;
    void onRightMouseButtonReleased() override;

private:
    void init() override {}
    void process() override;
    void shutdown() override {}

    void selectObjects(RealVector2D const& viewPos, bool modifierKeyPressed);
    void fixateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& worldPosOnClick, RealVector2D const& selectionPositionOnClick);
    void accelerateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);
    void updateSelectionRect(RealRect const& rect);

    std::optional<RealVector2D> _worldPosOnClick;
    std::optional<RealVector2D> _selectionPositionOnClick;
    std::optional<RealRect> _selectionRect;
};
