#pragma once

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

    bool areInspectionWindowsActive() const;
    void onCloseAllInspectorWindows();

    bool isObjectInspectionPossible() const;
    bool isGenomeInspectionPossible() const;
    bool isCreatureInspectionPossible() const;
    void onInspectSelectedObjects();
    void onInspectSelectedGenomes();
    void onInspectSelectedCreatures();
    bool onInspectObjects(std::vector<ExtendedObjectOrEnergyDesc> const& entities, bool creatureMode);

    bool isCopyingPossible() const;
    void onCopy();
    bool isPastingPossible() const;
    void onPaste();
    bool isDeletingPossible() const;
    void onDelete();

    void onOpenPattern();
    bool isSavingPatternPossible() const;
    void onSavePattern();

    void onColorSelectedObjects(int color);
    void onSetSticky(bool value);
    void onSetFixed(bool value);
    void onUniformVelocities();
    void onReleaseStresses();
    void onGlueSelectedObjects();

    void onSelectObjects(RealVector2D const& viewPos, bool modifierKeyPressed);
    void onMoveSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);
    void onMoveSelectedObjectsBy(RealVector2D const& delta);
    void onRotateSelectedObjects(float angleDelta);
    void onSetVelocityOfSelectedObjects(RealVector2D const& velocity);
    void onSetAngularVelocityOfSelectedObjects(float angularVelocity);
    void onFixateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& initialViewPos, RealVector2D const& selectionPositionOnClick);
    void onUpdateSelectionRect(RealRect const& rect);
    void onApplyForces(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);
    void onAccelerateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);
    void onCutConnections(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);

private:
    void init() override;
    void process() override;
    void shutdown() override;

    void processInspectorWindows();

    bool _on = false;

    std::vector<InspectionWindow> _inspectorWindows;
    std::optional<ContentDesc> _copiedSelection;
    std::string _patternStartingPath;
};
