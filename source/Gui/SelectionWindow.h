#pragma once

#include "Definitions.h"
#include "EngineImpl/Definitions.h"

class _SelectionWindow
{
public:
    _SelectionWindow(EditorModel const& editorModel);
    ~_SelectionWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    EditorModel _editorModel; 

    bool _on = false;
};