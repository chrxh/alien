#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _GpuSettingsWindow
{
public:
    _GpuSettingsWindow(
        StyleRepository const& styleRepository,
        SimulationController const& simController,
        GlobalSettings const& globalSettings);

    ~_GpuSettingsWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    StyleRepository _styleRepository;
    SimulationController _simController;
    GlobalSettings _globalSettings;

    bool _on = false;
};