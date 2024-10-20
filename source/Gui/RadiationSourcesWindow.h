#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"

class RadiationSourcesWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(RadiationSourcesWindow);

private:
    RadiationSourcesWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void processIntern() override;

    bool processTab(int index); //returns false if tab should be closed
    void onAppendTab();
    void onDeleteTab(int index);

    RadiationSource createParticleSource() const;

    void validationAndCorrection(RadiationSource& source) const;

    SimulationFacade _simulationFacade;
};