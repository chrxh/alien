#pragma once

#include "EngineInterface/SimulationController.h"

#include "Definitions.h"

enum class CreationMode
{
    CreateParticle,
    CreateCell,
    CreateRect,
    CreateHexagon,
    CreateDisc
};

class _CreatorWindow
{
public:
    _CreatorWindow(SimulationController const& simController, Viewport const& viewport);
    ~_CreatorWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    void createCell();

    bool _on = false;

    float _energy = 100.0f;
    float _distance = 1.0f;

    CreationMode _mode = CreationMode::CreateCell;

    SimulationController _simController;
    Viewport _viewport;
};
