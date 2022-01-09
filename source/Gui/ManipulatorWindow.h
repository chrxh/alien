#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/Descriptions.h"
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

    bool isInspectionPossible() const;
    void onInspectEntities();

private:
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
    std::optional<SelectionShallowData> _lastSelection;
    std::optional<DataDescription> _copiedSelection;
};