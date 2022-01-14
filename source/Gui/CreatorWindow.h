#pragma once

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
    _CreatorWindow();
    ~_CreatorWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    bool _on = false;

    float _energy = 100.0f;
    float _distance = 1.0f;

    CreationMode _mode = CreationMode::CreateCell;
};
