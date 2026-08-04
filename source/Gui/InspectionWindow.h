#pragma once

#include <optional>
#include <string>
#include <vector>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Descs.h>

#include "Definitions.h"

class _InspectionWindow
{
public:
    _InspectionWindow(uint64_t entityId, RealVector2D const& initialPos, bool creatureMode);
    ~_InspectionWindow();

    void process();

    bool isClosed() const;
    uint64_t getId() const;

private:
    bool isExtendedObject() const;
    std::string generateTitle() const;

    void processObject(ExtendedObjectDesc& extendedObject);
    void processCreature(ExtendedObjectDesc& extendedObject);
    void processParticle(EnergyDesc particle);
    void applyPendingSignalEntries(ExtendedObjectDesc& extendedObject);

    void processObjectNode(ObjectDesc& object);
    void processSolidNode(ObjectDesc& object);
    void processFluidNode(ObjectDesc& object);
    void processFreeCellNode(ObjectDesc& object);
    void processCellNode(ObjectDesc& object);
    void processCreatureNode(ExtendedObjectDesc& extendedObject);
    void processCreatureProperties(ExtendedObjectDesc& extendedObject);
    void processNeuralActivityNode(CellDesc& cell);
    void processNeuralNetNode(ObjectDesc& object);
    void processCellTypeNode(CellDesc& cell);
    void processConstructorNode(ConstructorDesc& constructor);

    float calcWindowWidth() const;

    RealVector2D _initialPos;

    bool _on = true;
    uint64_t _entityId = 0;
    bool _creatureMode = false;
    bool _isFirstFrame = true;
    std::optional<std::vector<SignalEntryDesc>> _pendingSignalEntries;

    NeuralNetEditorWidget _neuralNetWidget;

    static std::optional<float> _savedScrollY;
};
