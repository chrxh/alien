#pragma once

#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/Descriptions.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _ManipulatorWindow
{
public:
    _ManipulatorWindow(
        EditorModel const& editorModel,
        SimulationController const& simController,
        Viewport const& viewport);
    ~_ManipulatorWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    bool nexusButton();
    bool colorButton(std::string id, uint32_t cellColor);
    bool hasSelectionChanged(SelectionShallowData const& selection) const;

    EditorModel _editorModel;
    SimulationController _simController;
    Viewport _viewport;
    OpenSelectionDialog _openSelectionDialog;
    SaveSelectionDialog _saveSelectionDialog;

    bool _on = false;
    bool _includeClusters = true;
    float _angle = 0;
    float _angularVel = 0;
    boost::optional<SelectionShallowData> _lastSelection;
    boost::optional<DataDescription> _copiedSelection;
};