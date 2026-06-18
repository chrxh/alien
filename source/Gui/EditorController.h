#pragma once

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <EngineInterface/Desc.h>
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

    void onSelectObjects(RealVector2D const& viewPos, bool modifierKeyPressed);
    void onMoveSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);
    void onFixateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& initialViewPos, RealVector2D const& selectionPositionOnClick);
    void onUpdateSelectionRect(RealRect const& rect);
    void onApplyForces(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);
    void onAccelerateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);

private:
    void init() override;
    void process() override;
    void shutdown() override {}

    void processInspectorWindows();

    bool _on = false;

    std::vector<InspectionWindow> _inspectorWindows;
    Desc _drawing;
};
