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
        Editor
    };
    Mode getMode() const;
    void setMode(Mode value);

private:
    EditorController _editorController;

    TextureData _editorOn;
    TextureData _editorOff;
    
    Mode _mode = Mode::Navigation;
};
