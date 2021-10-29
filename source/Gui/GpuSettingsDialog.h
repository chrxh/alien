#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _GpuSettingsDialog
{
public:
    _GpuSettingsDialog(
        StyleRepository const& styleRepository,
        SimulationController const& simController);

    ~_GpuSettingsDialog();

    void process();

    void show();

private:
    StyleRepository _styleRepository;
    SimulationController _simController;

    bool _show = false;
    GpuSettings _gpuSettings;
};