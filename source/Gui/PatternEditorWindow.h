#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class PatternEditorWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(PatternEditorWindow);

public:
    bool isObjectInspectionPossible() const;
    bool isGenomeInspectionPossible() const;

    bool isCopyingPossible() const;
    void onCopy();
    bool isPastingPossible() const;
    void onPaste();
    bool isDeletingPossible() const;
    void onDelete();

private:
    PatternEditorWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;
    bool isShown() override;

    void onOpenPattern();
    void onSavePattern();
    void onMakeSticky();
    void onRemoveStickiness();
    void onSetBarrier(bool value);
    bool colorButton(std::string id, uint32_t cellColor);
    bool hasSelectionChanged(SelectionShallowData const& selection) const;

    SimulationFacade _simulationFacade;

    std::string _startingPath;
    float _angle = 0;
    float _angularVel = 0;
    std::optional<SelectionShallowData> _lastSelection;
    std::optional<CollectionDescription> _copiedSelection;
};