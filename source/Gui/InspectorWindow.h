#pragma once

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Description.h>

#include "Definitions.h"

struct MemoryEditor;
struct CompilationResult;

class _InspectorWindow
{
public:
    _InspectorWindow(uint64_t entityId, RealVector2D const& initialPos, bool selectGenomeTab);
    ~_InspectorWindow();

    void process();

    bool isClosed() const;
    uint64_t getId() const;

private:
    bool isCell() const;
    std::string generateTitle() const;

    void processCell(ExtendedObjectDesc& creatureCell);
    void processCellGeneralTab(ExtendedObjectDesc& creatureCell);
    void processCellTypeTab(ObjectDesc& cell);
    void processCellTypePropertiesTab(ObjectDesc& cell);
    template <typename Desc>
    void processCellGenomeTab(Desc& desc);

    void processGeneratorContent(GeneratorDesc& generator);
    void processNeuronContent(ObjectDesc& cell);
    void processConstructorContent(ConstructorDesc& constructor);
    void processInjectorContent(InjectorDesc& injector);
    void processAttackerContent(AttackerDesc& attacker);
    void processDefenderContent(DefenderDesc& defender);
    void processDepotContent(DepotDesc& transmitter);
    void processMuscleContent(MuscleDesc& muscle);
    void processSensorContent(SensorDesc& sensor);
    void processReconnectorContent(ReconnectorDesc& reconnector);
    void processDetonatorContent(DetonatorDesc& detonator);

    void processParticle(EnergyDesc particle);

    float calcWindowWidth() const;

    void validateAndCorrect(ObjectDesc& cell) const;

    RealVector2D _initialPos;

    bool _on = true;
    uint64_t _entityId = 0;
    float _genomeZoom = 20.0f;
    bool _selectGenomeTab = false;
};
