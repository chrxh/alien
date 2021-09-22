#pragma once

#include "Definitions.h"

class _ModeWindow
{
public:
    _ModeWindow();

    void process();

private:
    TextureData _navigationOn;
    TextureData _navigationOff;
    TextureData _actionOn;
    TextureData _actionOff;
    
    enum Mode
    {
        Navigation,
        Action
    };
    Mode _mode = Mode::Navigation;
};
