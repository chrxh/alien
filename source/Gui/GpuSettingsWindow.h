#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _GpuSettingsWindow
{
public:
    _GpuSettingsWindow(
        StyleRepository const& styleRepository,
        SimulationController const& simController);

    ~_GpuSettingsWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    StyleRepository _styleRepository;
    SimulationController _simController;

    bool _on = false;
};