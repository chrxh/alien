#pragma once

#include "EngineInterface/Descriptions.h"
#include "Definitions.h"

class _InspectorWindow
{
public:
    _InspectorWindow(
        Viewport const& viewport,
        EditorModel const& editorModel,
        uint64_t entityId,
        RealVector2D const& initialPos);
    ~_InspectorWindow();

    void process();

    bool isClosed() const;
    uint64_t getId() const;

private:
    std::string generateTitle() const;
    
private:
    Viewport _viewport; 
    EditorModel _editorModel;
    RealVector2D _initialPos;

    bool _on = true;
    uint64_t _entityId = 0;
};
