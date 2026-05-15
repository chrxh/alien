#pragma once

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Desc.h>

#include "Definitions.h"

class _InspectionWindow
{
public:
    _InspectionWindow(uint64_t entityId, RealVector2D const& initialPos, bool selectGenomeTab);
    ~_InspectionWindow();

    void process();

    bool isClosed() const;
    uint64_t getId() const;

private:
    bool isObject() const;
    std::string generateTitle() const;

    void processObject(ExtendedObjectDesc& extendedObject);
    void processParticle(EnergyDesc particle);

    void processObjectNode(ObjectDesc& object);
    void processSolidNode(ObjectDesc& object);
    void processFluidNode(ObjectDesc& object);
    void processFreeCellNode(ObjectDesc& object);
    void processCellNode(ObjectDesc& object);
    void processCreatureNode(ExtendedObjectDesc& extendedObject);
    void processSignalsNode(CellDesc& cell);
    void processNeuralNetNode(CellDesc& cell);
    void processCellTypeNode(CellDesc& cell);
    void processConstructorSubNode(ConstructorDesc& constructor);

    float calcWindowWidth() const;

    void validateAndCorrect(ObjectDesc& object) const;

    RealVector2D _initialPos;

    bool _on = true;
    uint64_t _entityId = 0;
    bool _selectGenomeTab = false;

    NeuralNetEditorWidget _neuralNetWidget;
};
