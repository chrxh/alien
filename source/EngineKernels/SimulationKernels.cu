#include "AttackerProcessor.cuh"
#include "ClusterProcessor.cuh"
#include "CommunicatorProcessor.cuh"
#include "ConstructorProcessor.cuh"
#include "DepotProcessor.cuh"
#include "DetonatorProcessor.cuh"
#include "DigestorProcessor.cuh"
#include "ForceFieldKernels.cuh"
#include "GeneratorProcessor.cuh"
#include "InjectorProcessor.cuh"
#include "MemoryProcessor.cuh"
#include "MutationProcessor.cuh"
#include "ObjectProcessor.cuh"
#include "ReconnectorProcessor.cuh"
#include "SensorProcessor.cuh"
#include "SimulationKernels.cuh"
#include "VoidProcessor.cuh"

__global__ void cudaNextTimestep_prepare(SimulationData data)
{
    data.objectMap.reset();
    data.energyMap.reset();
    data.processMemory.reset();

    // Heuristics
    auto maxStructureOperations = 1000 + data.entities.objects.getNumEntries() / 2;
    auto maxCellTypeOperations = data.entities.objects.getNumEntries();

    data.structuralOperations.setMemory(data.processMemory.getTypedSubArray<StructuralOperation>(maxStructureOperations), maxStructureOperations);

    for (int i = CellType_Base; i < CellType_Count; ++i) {
        data.cellTypeOperations[i].setMemory(data.processMemory.getTypedSubArray<CellTypeOperation>(maxCellTypeOperations), maxCellTypeOperations);
    }
    *data.externalEnergy = cudaSimulationParameters.externalEnergy.value;
    for (int i = 0; i < MAX_COLORS; ++i) {
        data.numConstructorsNeedingEnergyByColor[i] = 0;
        data.externalEnergyInflowPerConstructorByColor[i] = 0.0f;
    }

    data.entities.saveNumEntries();
}

__global__ void cudaNextTimestep_physics_init(SimulationData data)
{
    ObjectProcessor::init(data);
    EnergyProcessor::calcActiveSources(data);
}

__global__ void cudaNextTimestep_physics_fillMaps(SimulationData data)
{
    ObjectProcessor::updateMap(data);
    ObjectProcessor::radiation(data);  //do not use EnergyParticleProcessor in this calcKernel
    ObjectProcessor::clearDensityMap(data);
}

__global__ void cudaNextTimestep_physics_calcFluidForces(SimulationData data)
{
    ObjectProcessor::calcFluidForces_reconnectCells_correctOverlap(data);
    ObjectProcessor::fillDensityMap(data);
    EnergyProcessor::fillDensityMap(data);

    EnergyProcessor::updateMap(data);
}

__global__ void cudaNextTimestep_physics_calcFluidBoundaryForces(SimulationData data)
{
    ObjectProcessor::calcFluidBoundaryForces(data);
}

__global__ void cudaNextTimestep_physics_applyForces(SimulationData data)
{
    ObjectProcessor::checkForces(data);
    ObjectProcessor::applyForces(data);

    EnergyProcessor::movement(data);
    EnergyProcessor::collision(data);
}

__global__ void cudaNextTimestep_physics_verletPositionUpdate(SimulationData data)
{
    ObjectProcessor::verletPositionUpdate(data);
    ObjectProcessor::checkConnections(data);

    EnergyProcessor::splitting(data);
}

__global__ void cudaNextTimestep_physics_calcConnectionForces(SimulationData data, bool considerAngles)
{
    ObjectProcessor::calcConnectionForces(data, considerAngles);
}

__global__ void cudaNextTimestep_physics_verletVelocityUpdate(SimulationData data)
{
    ObjectProcessor::verletVelocityUpdate(data);
}

__global__ void cudaNextTimestep_signal_calcSignal(SimulationData data, SimulationStatistics statistics)
{
    NeuronProcessor::calcSignal(data, statistics);
}

__global__ void cudaNextTimestep_signal_setSignal(SimulationData data)
{
    NeuronProcessor::setSignal(data);
}

__global__ void cudaNextTimestep_energyFlow(SimulationData data)
{
    CellProcessor::performEnergyFlow(data);
}

__global__ void cudaNextTimestep_cellState_substep1(SimulationData data)
{
    CellProcessor::aging(data);
    CellProcessor::cellStateTransition_calcFutureState(data);
    CellProcessor::headUpdate_calcFutureValue(data);
}

__global__ void cudaNextTimestep_cellState_substep2(SimulationData data)
{
    CellProcessor::cellStateTransition_applyNextState(data);
    CellProcessor::headUpdate_applyFutureValue(data);
}

__global__ void cudaNextTimestep_cellType_prepare_substep1(SimulationData data)
{
    CellProcessor::collectCellTypeOperations(data);
    CellProcessor::updateCellEvents(data);
}

__global__ void cudaNextTimestep_cellType_generator(SimulationData data, SimulationStatistics statistics)
{
    GeneratorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_constructor(SimulationData data, SimulationStatistics statistics, bool isPreview)
{
    ConstructorProcessor::process(data, statistics, isPreview);
}

__global__ void cudaNextTimestep_constructor_countConstructorsNeedingEnergy(SimulationData data)
{
    ConstructorProcessor::countConstructorsNeedingEnergy(data);
}

__global__ void cudaNextTimestep_constructor_prepareExternalEnergyInflow(SimulationData data)
{
    auto totalEnergyNeeded = 0.0;
    for (int color = 0; color < MAX_COLORS; ++color) {
        totalEnergyNeeded += data.numConstructorsNeedingEnergyByColor[color] * 50.0 * cudaSimulationParameters.externalEnergyInflowFactor.value[color];
    }
    auto externalEnergy = alienAtomicRead(data.externalEnergy);
    auto factor = 0.0;
    if (totalEnergyNeeded > 0.0 && externalEnergy > 0.0) {
        factor = externalEnergy == Infinity<float>::value ? 1.0 : min(1.0, externalEnergy / totalEnergyNeeded);
    }
    for (int color = 0; color < MAX_COLORS; ++color) {
        data.externalEnergyInflowPerConstructorByColor[color] = 50.0f * cudaSimulationParameters.externalEnergyInflowFactor.value[color] * factor;
    }
    if (factor > 0.0 && externalEnergy != Infinity<float>::value) {
        *data.externalEnergy = max(0.0, externalEnergy - totalEnergyNeeded);
    }
}

__global__ void cudaNextTimestep_constructor_provideExternalEnergy(SimulationData data)
{
    ConstructorProcessor::provideExternalEnergy(data);
}

__global__ void cudaNextTimestep_applyMutations(SimulationData data, SimulationStatistics statistics)
{
    MutationProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_injector(SimulationData data, SimulationStatistics statistics)
{
    InjectorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_attacker(SimulationData data, SimulationStatistics statistics)
{
    AttackerProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_depot(SimulationData data, SimulationStatistics statistics)
{
    DepotProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_muscle(SimulationData data, SimulationStatistics statistics)
{
    MuscleProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_sensor(SimulationData data, SimulationStatistics statistics)
{
    SensorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_reconnector(SimulationData data, SimulationStatistics statistics)
{
    ReconnectorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_detonator(SimulationData data, SimulationStatistics statistics)
{
    DetonatorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_digestor(SimulationData data, SimulationStatistics statistics)
{
    DigestorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_memory(SimulationData data, SimulationStatistics statistics)
{
    MemoryProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_communicator(SimulationData data, SimulationStatistics statistics)
{
    CommunicatorProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_cellType_void(SimulationData data, SimulationStatistics statistics)
{
    VoidProcessor::process(data, statistics);
}

__global__ void cudaNextTimestep_physics_applyInnerFriction(SimulationData data)
{
    ObjectProcessor::applyInnerFriction(data);
}

__global__ void cudaNextTimestep_physics_applyFriction(SimulationData data, bool isPreview)
{
    ObjectProcessor::applyFriction(data);
    CellProcessor::decay(data, isPreview);
}

__global__ void cudaNextTimestep_structuralOperations_substep1(SimulationData data)
{
    ObjectConnectionProcessor::processAddOperations(data);
    ObjectConnectionProcessor::processDeleteConnectionObjectOperations_prepare(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep2(SimulationData data)
{
    ObjectConnectionProcessor::processDeleteConnectionObjectOperations_step1(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep3(SimulationData data)
{
    ObjectConnectionProcessor::processDeleteConnectionObjectOperations_step2(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep4(SimulationData data)
{
    ObjectConnectionProcessor::processDeleteObjectOperations(data);
}

__global__ void cudaNextTimestep_structuralOperations_substep5(SimulationData data)
{
    EnergyProcessor::transformation(data);
}

__global__ void cudaNextTimestep_incTimestep(SimulationData data)
{
    ++(*data.timestep);
}

__global__ void cudaInitClusterData(SimulationData data)
{
    ClusterProcessor::initClusterData(data);
}

__global__ void cudaFindClusterIteration(SimulationData data)
{
    ClusterProcessor::findClusterIteration(data);
}

__global__ void cudaFindClusterBoundaries(SimulationData data)
{
    ClusterProcessor::findClusterBoundaries(data);
}

__global__ void cudaAccumulateClusterPosAndVel(SimulationData data)
{
    ClusterProcessor::accumulateClusterPosAndVel(data);
}

__global__ void cudaAccumulateClusterAngularProp(SimulationData data)
{
    ClusterProcessor::accumulateClusterAngularProp(data);
}

__global__ void cudaApplyClusterData(SimulationData data)
{
    ClusterProcessor::applyClusterData(data);
}

__global__ void cudaResetDensity(SimulationData data)
{
    ObjectProcessor::resetDensity(data);
}
