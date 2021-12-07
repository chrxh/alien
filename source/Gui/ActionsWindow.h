#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _ActionsWindow
{
public:
    _ActionsWindow(
        EditorModel const& editorModel,
        SimulationController const& simController,
        StyleRepository const& styleRepository);
    ~_ActionsWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    EditorModel _editorModel;
    SimulationController _simController;
    StyleRepository _styleRepository;

    bool _on = false;
    bool _includeClusters = true;
    float _angle = 0;
};