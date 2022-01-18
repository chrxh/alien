#pragma once

#include "Definitions.h"

class _ModeController
{
public:
    _ModeController(EditorController const& editorController);

    void process();

    enum class Mode
    {
        Navigation,
        Action
    };
    Mode getMode() const;
    void setMode(Mode value);

private:
    EditorController _editorController;

    TextureData _navigationOn;
    TextureData _navigationOff;
    TextureData _actionOn;
    TextureData _actionOff;
    
    Mode _mode = Mode::Navigation;
};
