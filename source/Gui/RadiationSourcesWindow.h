#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"

class RadiationSourcesWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(RadiationSourcesWindow);

public:
    void init(SimulationFacade const& simulationFacade);

private:
    RadiationSourcesWindow();

    void processIntern() override;

    bool processTab(int index); //returns false if tab should be closed
    void onAppendTab();
    void onDeleteTab(int index);

    RadiationSource createParticleSource() const;

    void validationAndCorrection(RadiationSource& source) const;

    SimulationFacade _simulationFacade;
};