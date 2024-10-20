#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/GpuSettings.h"

#include "AlienDialog.h"
#include "Definitions.h"

class GpuSettingsDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GpuSettingsDialog);

public:
    void init(SimulationFacade const& simulationFacade);
    void shutdown();

private:
    GpuSettingsDialog();

    void processIntern();
    void openIntern();

    void validationAndCorrection(GpuSettings& settings) const;

    SimulationFacade _simulationFacade;

    GpuSettings _gpuSettings;
};