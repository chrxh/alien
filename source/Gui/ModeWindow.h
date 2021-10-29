#pragma once

#include "Definitions.h"

class _ModeWindow
{
public:
    _ModeWindow();

    void process();

    enum class Mode
    {
        Navigation,
        Action
    };
    Mode getMode() const;

private:
    TextureData _navigationOn;
    TextureData _navigationOff;
    TextureData _actionOn;
    TextureData _actionOff;
    
    Mode _mode = Mode::Navigation;
};
