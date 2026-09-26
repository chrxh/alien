#pragma once

#include <Base/Singleton.h>

#include <Data/Descs.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class InspectionController : public MainLoopEntity
{
    MAKE_SINGLETON(InspectionController);

public:
    bool areInspectionWindowsActive() const;
    void onCloseAllInspectorWindows();

    bool isObjectInspectionPossible() const;
    bool isGenomeInspectionPossible() const;
    bool isCreatureInspectionPossible() const;
    void onInspectSelectedObjects();
    void onInspectSelectedGenomes();
    void onInspectSelectedCreatures();

private:
    void init() override {}
    void process() override;
    void shutdown() override {}

    bool inspectObjects(std::vector<ExtendedObjectOrEnergyDesc> const& entities, bool creatureMode);

    std::vector<InspectionWindow> _inspectorWindows;
};
