#pragma once

#include "Definitions.h"

class _ActionsWindow
{
public:
    _ActionsWindow(
        EditorModel const& editorModel,
        StyleRepository const& styleRepository);
    ~_ActionsWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    EditorModel _editorModel;
    StyleRepository _styleRepository;

    bool _on = false;
    bool _includeClusters = true;
};