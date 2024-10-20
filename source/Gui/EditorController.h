#pragma once

#include "Base/Definitions.h"
#include "Base/Singleton.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class EditorController
{
    MAKE_SINGLETON(EditorController);

public:
    void init(SimulationFacade const& simulationFacade);

    bool isOn() const;
    void setOn(bool value);

    void process();

    bool areInspectionWindowsActive() const;
    void onCloseAllInspectorWindows();

    void onInspectSelectedObjects();
    void onInspectSelectedGenomes();
    void onInspectObjects(std::vector<CellOrParticleDescription> const& entities, bool selectGenomeTab);

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
    void processInspectorWindows();

    SimulationFacade _simulationFacade;

    bool _on = false;   //#TODO weg!

    std::vector<InspectorWindow> _inspectorWindows;
    DataDescription _drawing;
};
