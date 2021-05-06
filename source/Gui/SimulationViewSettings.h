#pragma once

struct SimulationViewSettings
{
    bool glowEffect = true;
    bool motionEffect = true;

    enum class Mode
    {
        NavigationMode,
        ActionMode
    };
    Mode mode = Mode::NavigationMode;
};