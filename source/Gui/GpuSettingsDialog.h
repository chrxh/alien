#pragma once

#include "AlienDialog.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/GpuSettings.h"
#include "Definitions.h"

class _GpuSettingsDialog : public _AlienDialog
{
public:
    _GpuSettingsDialog(SimulationController const& simController);

    ~_GpuSettingsDialog();

private:
    void processIntern();
    void openIntern();

    void validationAndCorrection(GpuSettings& settings) const;

    SimulationController _simController;

    GpuSettings _gpuSettings;
};