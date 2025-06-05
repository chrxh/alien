#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class CreatureEditorWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(CreatureEditorWindow);

public:

private:
    CreatureEditorWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;
    bool isShown() override;

    void processToolbar();
    void processTabWidget();

    struct TempLayoutData
    {
        float origGenomeEditorWidth;
    };
    TempLayoutData beginCorrectingLayout();
    void endCorrectingLayout(TempLayoutData const& tempLayoutData);

    SimulationFacade _simulationFacade;

    // Layout data
    CreatureTabLayoutData _creatureTabLayoutData;
    std::optional<RealVector2D> _lastWindowSize;

    // Tab data
    std::vector<CreatureTabWidget> _tabs;
    int _selectedTabIndex = 0;
    //std::optional<TabData> _tabToAdd;
};
