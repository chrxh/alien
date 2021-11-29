#pragma once

#include "Definitions.h"
#include "EngineImpl/Definitions.h"

class _SelectionWindow
{
public:
    _SelectionWindow(EditorModel const& editorModel, StyleRepository const& styleRepository);
    ~_SelectionWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    EditorModel _editorModel; 
    StyleRepository _styleRepository;

    bool _on = false;
};