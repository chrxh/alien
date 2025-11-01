#pragma once

#include <vector>
#include <optional>
#include <type_traits>

#include "Base/Singleton.h"

#include "Description.h"
#include "GenomeDescription.h"
#include "SimulationParameters.h"

using GeneIndicesForSubGenome = std::vector<int>;
class GenomeDescriptionEditService
{
    MAKE_SINGLETON(GenomeDescriptionEditService);

public:
    void addGene(GenomeDescription& genome, int index, GeneDescription const& newGene) const;  // Adds gene after index
    void removeGene(GenomeDescription& genome, int index) const;
    void swapGenes(GenomeDescription& genome, int index) const;  // Swaps gene at index with gene at index + 1

    void addNode(GeneDescription& gene, int index, NodeDescription const& node) const;  // Adds node after index
    void removeNode(GeneDescription& gene, int index) const;
    void swapNodes(GeneDescription& gene, int index) const;  // Swaps node at index with node at index + 1

    std::vector<SubGenomeDescription> createSubGenomesForPreview(
        GenomeDescription const& genome,
        std::vector<GeneIndicesForSubGenome> const& geneIndicesForSubGenomes,
        bool detailSimulation) const;

    struct SeedCollectionResult
    {
        Description description; 
        std::vector<uint64_t> seedCreatureIds;
    };
    
    template<typename CacheType>
    SeedCollectionResult createSeedCollectionForPreview(
        std::vector<SubGenomeDescription> const& subGenomes,
        CacheType const& cache) const;
        
    std::vector<Description> extractPhenotypesFromPreview(Description&& preview, std::vector<uint64_t> const& seedCreatureIds) const;
    void removeSeedFromPhenotype(Description& phenotype) const;

private:
    Description createSeedForPreview(SubGenomeDescription const& subGenome, RealVector2D const& pos) const;

    void adaptDescriptionForPreview(GenomeDescription& genome, GeneIndicesForSubGenome const& geneIndices, bool detailSimulation) const;
};

// Template implementation needs to be in header
#include "DescriptionEditService.h"
#include "EngineConstants.h"

namespace
{
    // Helper to detect if cache.find() returns an optional
    template<typename T>
    struct returns_optional_from_find
    {
    private:
        template<typename U>
        static auto test(int) -> typename std::is_same<
            std::optional<Description>,
            decltype(std::declval<U const&>().find(std::declval<SubGenomeDescription const&>()))
        >::type;
        
        template<typename>
        static std::false_type test(...);
        
    public:
        static constexpr bool value = decltype(test<T>(0))::value;
    };
}

template<typename CacheType>
auto GenomeDescriptionEditService::createSeedCollectionForPreview(
    std::vector<SubGenomeDescription> const& subGenomes,
    CacheType const& cache) const -> SeedCollectionResult
{
    auto const& editService = DescriptionEditService::get();

    RealVector2D currentPos{toFloat(PREVIEW_HEIGHT) / 2, toFloat(PREVIEW_HEIGHT) / 2};

    SeedCollectionResult result;
    for (auto const& subGenome : subGenomes) {
        std::optional<Description> cachedValue;
        
        // Handle both Cache (returns optional) and unordered_map (returns iterator)
        if constexpr (returns_optional_from_find<CacheType>::value) {
            // Cache API: find returns std::optional<Value>
            cachedValue = cache.find(subGenome);
        } else {
            // unordered_map API: find returns iterator
            auto findResult = cache.find(subGenome);
            if (findResult != cache.end()) {
                cachedValue = findResult->second;
            }
        }
        
        if (cachedValue.has_value()) {
            auto cachedPhenotype = cachedValue.value();
            editService.setCenter(cachedPhenotype, currentPos);

            CHECK(cachedPhenotype._creatures.size() <= 2);
            auto seedFirst = false;
            if (cachedPhenotype._creatures.front()._generation == 0) {
                seedFirst = true;  // first Creature is seed
            }
            result.description.add(std::move(cachedPhenotype));

            auto index =
                seedFirst ? result.description._creatures.size() - cachedPhenotype._creatures.size() : result.description._creatures.size() - cachedPhenotype._creatures.size() + 1;
            result.seedCreatureIds.emplace_back(result.description._creatures.at(index)._id);
        } else {
            auto seed = createSeedForPreview(subGenome, currentPos);
            result.description.add(std::move(seed));

            result.seedCreatureIds.emplace_back(result.description._creatures.back()._id);
        }
        currentPos.x += toFloat(PREVIEW_HEIGHT) / 2;  // Adjust position for the next sub-genome
    }
    return result;
}
