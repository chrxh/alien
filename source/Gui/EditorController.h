#pragma once

#include <chrono>

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <Data/Descs.h>

#include <EngineInterface/SimulationFacade.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class EditorController : public MainLoopEntity
{
    MAKE_SINGLETON(EditorController);

public:
    bool isOn() const;
    void setOn(bool value);

    bool isCopyingPossible() const;
    void onCopy();
    bool isPastingPossible() const;
    void onPaste();
    bool isDeletingPossible() const;
    void onDelete();
    bool isDeselectingPossible() const;
    void onDeselect();

    void onColorSelectedObjects(int color);
    void onSetSticky(bool value);
    void onSetStatic(bool value);
    void onUniformVelocities();
    void onReleaseStresses();
    void onGlueSelectedObjects();

    void onMoveSelectedObjectsBy(RealVector2D const& delta);
    void onRotateSelectedObjects(float angleDelta);
    void onSetVelocityOfSelectedObjects(RealVector2D const& velocity);
    void onSetAngularVelocityOfSelectedObjects(float angularVelocity);

private:
    void init() override;
    void process() override;
    void shutdown() override;

    bool _on = false;

    std::optional<ContentDesc> _copiedSelection;
    std::chrono::steady_clock::time_point _lastSelectionRolloutTime;
};
