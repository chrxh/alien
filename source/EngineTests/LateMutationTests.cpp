#include <unordered_map>
#include <unordered_set>

#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/SimulationFacade.h>
#include <PersisterInterface/DeserializedSimulation.h>
#include <PersisterInterface/SerializerService.h>

#include "IntegrationTestFramework.h"

#ifndef TEST_RESOURCE_DIR
#define TEST_RESOURCE_DIR "."
#endif

// World size of the captured latemutation.sim (see Resources/latemutation.settings.json).
class LateMutationTests : public IntegrationTestFramework
{
public:
    LateMutationTests()
        : IntegrationTestFramework({5000, 1500})
    {}

    ~LateMutationTests() override = default;
};

// Regression test for a mutation timing bug: a creature constructed an offspring before mutating itself.
// The constructor sets constructor.offspring on the first (energy-less) trigger, capturing the parent genome
// pointer. The host is then mutated later (its genome pointer is replaced by a freshly mutated clone), so the
// already-cloned offspring keeps the original, unmutated genome. Invariant that must always hold: a creature that
// has not been mutated yet (mutationState == NotMutated) must carry exactly the genome of its parent.
TEST_F(LateMutationTests, offspringInheritsMutatedGenome)
{
    DeserializedSimulation deserialized;
    ASSERT_TRUE(SerializerService::get().deserializeSimulationFromFiles(deserialized, std::string(TEST_RESOURCE_DIR) + "/latemutation.sim"));

    _simulationFacade->setSimulationParameters(deserialized.auxiliaryData.simulationParameters);
    _simulationFacade->setSimulationData(deserialized.mainData);

    // The .sim file was captured with the old (buggy) engine, so creatures already present at load may carry the
    // historical divergence. The fix can only repair creatures created from now on, so restrict the check to those.
    std::unordered_set<uint64_t> preExistingCreatureIds;
    for (auto const& creature : _simulationFacade->getSimulationData()._creatures) {
        preExistingCreatureIds.insert(creature._id);
    }

    auto checkInvariant = [&]() {
        auto data = _simulationFacade->getSimulationData();

        std::unordered_map<uint64_t, CreatureDesc> creatureById;
        for (auto const& creature : data._creatures) {
            creatureById.emplace(creature._id, creature);
        }
        std::unordered_map<uint64_t, GenomeDesc> genomeById;
        for (auto const& genome : data._genomes) {
            genomeById.emplace(genome._id, genome);
        }

        int checkedPairs = 0;
        for (auto const& offspring : data._creatures) {
            if (preExistingCreatureIds.count(offspring._id) != 0) {
                continue;
            }
            if (offspring._mutationState != MutationState_NotMutated || !offspring._ancestorId.has_value()) {
                continue;
            }
            auto parentIt = creatureById.find(*offspring._ancestorId);
            if (parentIt == creatureById.end()) {
                continue;
            }
            auto const& parent = parentIt->second;

            auto offspringGenomeIt = genomeById.find(offspring._genomeId);
            auto parentGenomeIt = genomeById.find(parent._genomeId);
            if (offspringGenomeIt == genomeById.end() || parentGenomeIt == genomeById.end()) {
                ADD_FAILURE() << "Missing genome for creature " << offspring._id << " or parent " << parent._id << ".";
                continue;
            }

            ++checkedPairs;
            EXPECT_TRUE(offspringGenomeIt->second.equalWithoutId(parentGenomeIt->second))
                << "Unmutated offspring creature " << offspring._id << " does not carry its parent's genome (parent " << parent._id << ").";
        }
        return checkedPairs;
    };

    int totalCheckedPairs = checkInvariant();
    for (int i = 0; i < 20; ++i) {
        _simulationFacade->calcTimesteps(50);
        totalCheckedPairs += checkInvariant();
    }

    EXPECT_GT(totalCheckedPairs, 0) << "Test is vacuous: no unmutated offspring with a living parent was observed.";
}
