#pragma once

#include "AlienDialog.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/GpuSettings.h"
#include "Definitions.h"

class _GpuSettingsDialog : public AlienDialog
{
public:
    _GpuSettingsDialog(SimulationFacade const& simulationFacade);
    ~_GpuSettingsDialog() override;

private:
    void processIntern();
    void openIntern();

    void validationAndCorrection(GpuSettings& settings) const;

    SimulationFacade _simulationFacade;

    GpuSettings _gpuSettings;
};