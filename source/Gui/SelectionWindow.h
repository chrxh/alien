#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"
#include "AlienWindow.h"

class _SelectionWindow : public AlienWindow
{
public:
    _SelectionWindow(EditorModel const& editorModel);

private:
    void processIntern();

    EditorModel _editorModel; 
};