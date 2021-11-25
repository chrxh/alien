#pragma once

#include "Definitions.h"
#include "EngineImpl/Definitions.h"
#include "EngineInterface/Descriptions.h"

class _SelectionWindow
{
public:
    _SelectionWindow(StyleRepository const& styleRepository);
    ~_SelectionWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

    struct SelectedEntities
    {
        int numCells;
        int numIndirectCells;
        int numParticles;
    };
    void setSelection(SelectedEntities const& selection);

private:
    StyleRepository _styleRepository;

    bool _on = false;
    SelectedEntities _selection = {0, 0};
};