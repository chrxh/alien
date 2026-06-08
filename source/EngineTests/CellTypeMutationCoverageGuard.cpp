// Compile-time guard for cell-type property mutation coverage.
//
// Each static_assert below pins the number of data members of
// cell-type or mode description. The matching list of mutated attributes
// lives in MutationProcessor::applyMutations_cellTypeProperties().
//
// When a field is added to or removed from one of these structs, the
// corresponding assert fails to compile. Treat that failure as a checklist:
// either extend the mutation switch for the new attribute and then bump the
// count here - or, if the field is intentionally never mutated, just bump the
// count. The point is to force a conscious decision so newly added cell-type
// attributes are never silently left out of the mutation.
//
// The mode counts at the bottom pin the number of modes per cell type. Modes
// themselves are not mutated by this mutation type, but a newly added mode
// brings its own description struct whose fields must be reviewed. When a mode
// is added or removed, the corresponding assert fails - extend the field-count
// list above and the mutation switch for the new mode, then bump the count here.


#include <variant>

#include <boost/pfr.hpp>

#include <EngineInterface/GenomeDesc.h>

#define ALIEN_MUTATION_FIELD_COUNT(Type, Count) \
    static_assert( \
        boost::pfr::tuple_size_v<Type> == (Count), \
        #Type " field count changed - review applyMutations_cellTypeProperties() in MutationProcessor.cuh and update this count")

#define ALIEN_MUTATION_MODE_COUNT(Type, Count) \
    static_assert( \
        std::variant_size_v<Type> == (Count), \
        #Type " mode count changed - review applyMutations_cellTypeProperties() in MutationProcessor.cuh and update this count")

// --- Cell types (switch (node.cellType)) ---
ALIEN_MUTATION_FIELD_COUNT(BaseGenomeDesc, 0);
ALIEN_MUTATION_FIELD_COUNT(DepotGenomeDesc, 2);
ALIEN_MUTATION_FIELD_COUNT(SensorGenomeDesc, 5);
ALIEN_MUTATION_FIELD_COUNT(GeneratorGenomeDesc, 5);
ALIEN_MUTATION_FIELD_COUNT(AttackerGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(InjectorGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(MuscleGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(DefenderGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(ReconnectorGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(DetonatorGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(DigestorGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(MemoryGenomeDesc, 3);
ALIEN_MUTATION_FIELD_COUNT(CommunicatorGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(VoidGenomeDesc, 0);

// --- Sensor modes (switch (node.cellTypeData.sensor.mode)) ---
ALIEN_MUTATION_FIELD_COUNT(TelemetryGenomeDesc, 0);
ALIEN_MUTATION_FIELD_COUNT(DetectEnergyGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(DetectSolidGenomeDesc, 0);
ALIEN_MUTATION_FIELD_COUNT(DetectFreeCellGenomeDesc, 2);
ALIEN_MUTATION_FIELD_COUNT(DetectCreatureGenomeDesc, 4);

// --- Generator modes (switch (node.cellTypeData.generator.mode)) ---
ALIEN_MUTATION_FIELD_COUNT(SquareSignalGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(SawtoothSignalGenomeDesc, 1);

// --- Attacker modes (switch (node.cellTypeData.attacker.mode)) ---
ALIEN_MUTATION_FIELD_COUNT(AttackFreeCellGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(AttackCreatureGenomeDesc, 0);

// --- Muscle modes (switch (node.cellTypeData.muscle.mode)) ---
ALIEN_MUTATION_FIELD_COUNT(AutoBendingGenomeDesc, 2);
ALIEN_MUTATION_FIELD_COUNT(ManualBendingGenomeDesc, 2);
ALIEN_MUTATION_FIELD_COUNT(AngleBendingGenomeDesc, 2);
ALIEN_MUTATION_FIELD_COUNT(AutoCrawlingGenomeDesc, 2);
ALIEN_MUTATION_FIELD_COUNT(ManualCrawlingGenomeDesc, 2);
ALIEN_MUTATION_FIELD_COUNT(DirectMovementGenomeDesc, 0);

// --- Reconnector modes (switch (node.cellTypeData.reconnector.mode)) ---
ALIEN_MUTATION_FIELD_COUNT(ReconnectSolidGenomeDesc, 0);
ALIEN_MUTATION_FIELD_COUNT(ReconnectFreeCellGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(ReconnectCreatureGenomeDesc, 4);

// --- Memory modes (switch (node.cellTypeData.memory.mode)) ---
ALIEN_MUTATION_FIELD_COUNT(SignalDelayGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(SignalRecorderGenomeDesc, 2);
ALIEN_MUTATION_FIELD_COUNT(SignalStorageGenomeDesc, 1);
ALIEN_MUTATION_FIELD_COUNT(SignalIntegratorGenomeDesc, 1);

// --- Communicator modes (switch (node.cellTypeData.communicator.mode)) ---
ALIEN_MUTATION_FIELD_COUNT(SenderGenomeDesc, 2);
ALIEN_MUTATION_FIELD_COUNT(ReceiverGenomeDesc, 2);

// --- Number of modes per cell type ---
ALIEN_MUTATION_MODE_COUNT(SensorModeGenomeDesc, 5);
ALIEN_MUTATION_MODE_COUNT(GeneratorModeGenomeDesc, 2);
ALIEN_MUTATION_MODE_COUNT(AttackerModeGenomeDesc, 2);
ALIEN_MUTATION_MODE_COUNT(MuscleModeGenomeDesc, 6);
ALIEN_MUTATION_MODE_COUNT(ReconnectorModeGenomeDesc, 3);
ALIEN_MUTATION_MODE_COUNT(MemoryModeGenomeDesc, 4);
ALIEN_MUTATION_MODE_COUNT(CommunicatorModeGenomeDesc, 2);

// Defender uses a plain enum for its mode instead of a variant.
static_assert(
    DefenderMode_Count == 2,
    "DefenderMode count changed - review applyMutations_cellTypeProperties() in MutationProcessor.cuh and update this count");

#undef ALIEN_MUTATION_MODE_COUNT
#undef ALIEN_MUTATION_FIELD_COUNT
