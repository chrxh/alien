#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "Definitions.h"

struct MemoryEditor;
struct CompilationResult;

class _InspectorWindow
{
public:
    _InspectorWindow(SimulationFacade const& simulationFacade, uint64_t entityId, RealVector2D const& initialPos, bool selectGenomeTab);
    ~_InspectorWindow();

    void process();

    bool isClosed() const;
    uint64_t getId() const;

private:
    bool isCell() const;
    std::string generateTitle() const;

    void processCell(CellDescription cell);
    void processCellBaseTab(CellDescription& cell);
    void processCellTypeTab(CellDescription& cell);
    void processCellTypePropertiesTab(CellDescription& cell);
    template <typename Description>
    void processCellGenomeTab(Description& desc);
    void processCellMetadataTab(CellDescription& cell);

    void processOscillatorContent(OscillatorDescription& oscillator);
    void processNeuronContent(CellDescription& cell);
    void processConstructorContent(ConstructorDescription& constructor);
    void processInjectorContent(InjectorDescription& injector);
    void processAttackerContent(AttackerDescription& attacker);
    void processDefenderContent(DefenderDescription& defender);
    void processTransmitterContent(DepotDescription& transmitter);
    void processMuscleContent(MuscleDescription& muscle);
    void processSensorContent(SensorDescription& sensor);
    void processReconnectorContent(ReconnectorDescription& reconnector);
    void processDetonatorContent(DetonatorDescription& detonator);

    void processParticle(ParticleDescription particle);

    float calcWindowWidth() const;

    void validateAndCorrect(CellDescription& cell) const;

    SimulationFacade _simulationFacade;

    RealVector2D _initialPos;

    bool _on = true;
    uint64_t _entityId = 0;
    float _genomeZoom = 20.0f;
    bool _selectGenomeTab = false;
};
