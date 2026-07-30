#pragma once

#include <cooperative_groups.h>

#include <EngineInterface/CellTypeConstants.h>

#include "SimulationData.cuh"

namespace cg_geneGraph = cooperative_groups;

// Operations on the directed graph formed by the genes of a genome, whose edges are the constructor references of their nodes.
class GeneGraphProcessor
{
public:
    __inline__ __device__ static void removeUnreachableGenesFromRoot(SimulationData& data, Genome* genome);
    __inline__ __device__ static void removeCyclesNotThroughRoot(SimulationData& data, Genome* genome);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void GeneGraphProcessor::removeUnreachableGenesFromRoot(SimulationData& data, Genome* genome)
{
    auto block = cg_geneGraph::this_thread_block();
    auto laneId = block.thread_rank();

    __shared__ int numGenes;
    __shared__ int newNumGenes;
    __shared__ int* newGeneIndices;  // During marking: -1 = unreachable, 1 = reachable but not scanned yet, 2 = scanned
    __shared__ bool anyGeneScanned;

    if (laneId == 0) {
        numGenes = genome->numGenes;
    }
    block.sync();

    if (laneId == 0) {
        newGeneIndices = data.entities.heap.getTypedSubArray<int>(numGenes);
    }
    block.sync();

    for (int geneIndex = laneId; geneIndex < numGenes; geneIndex += blockDim.x) {
        newGeneIndices[geneIndex] = geneIndex == 0 ? 1 : -1;
    }
    block.sync();

    // Compute the reachable set in parallel: each sweep scans the reachable genes found so far and marks the genes they
    // reference. The barrier between sweeps makes all marks visible; sweeps continue until a sweep scans no gene anymore.
    // Concurrent marking within a sweep is benign: a gene is scanned at least once and a redundant re-mark only costs an
    // extra sweep.
    do {
        if (laneId == 0) {
            anyGeneScanned = false;
        }
        block.sync();
        for (int geneIndex = laneId; geneIndex < numGenes; geneIndex += blockDim.x) {
            if (newGeneIndices[geneIndex] != 1) {
                continue;
            }
            newGeneIndices[geneIndex] = 2;
            anyGeneScanned = true;
            auto& gene = genome->genes[geneIndex];
            for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
                auto& node = gene.nodes[nodeIndex];
                if (node.constructorAvailable && newGeneIndices[node.constructor.geneIndex] == -1) {
                    newGeneIndices[node.constructor.geneIndex] = 1;
                }
            }
        }
        block.sync();
    } while (anyGeneScanned);

    // Assign compacted indices to the reachable genes.
    if (laneId == 0) {
        int nextIndex = 0;
        for (int geneIndex = 0; geneIndex < numGenes; ++geneIndex) {
            newGeneIndices[geneIndex] = newGeneIndices[geneIndex] >= 0 ? nextIndex++ : -1;
        }
        newNumGenes = nextIndex;
    }
    block.sync();

    if (newNumGenes != numGenes) {
        // Remap the references of the surviving genes in parallel; the reachable set is closed under constructor references, so
        // every constructor reference of a survivor maps to a survivor. Injector references are not part of the reachability
        // closure, so an injector may point at a removed gene; fall back to gene 0 in that case.
        for (int geneIndex = laneId; geneIndex < numGenes; geneIndex += blockDim.x) {
            if (newGeneIndices[geneIndex] < 0) {
                continue;
            }
            auto& gene = genome->genes[geneIndex];
            for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
                auto& node = gene.nodes[nodeIndex];
                if (node.constructorAvailable) {
                    node.constructor.geneIndex = newGeneIndices[node.constructor.geneIndex];
                }
                if (node.cellType == CellType_Injector) {
                    int mapped = newGeneIndices[node.cellTypeData.injector.geneIndex];
                    node.cellTypeData.injector.geneIndex = mapped < 0 ? 0 : mapped;
                }
            }
        }
        block.sync();

        // Compact the survivors within the existing gene array in parallel. A survivor only moves towards a smaller index and
        // the genes are processed in windows of blockDim.x with a barrier between reading and writing, so a gene is never
        // overwritten before it was read.
        for (int windowStart = 0; windowStart < numGenes; windowStart += blockDim.x) {
            auto geneIndex = windowStart + laneId;
            Gene movedGene;
            int targetIndex = -1;
            if (geneIndex < numGenes) {
                targetIndex = newGeneIndices[geneIndex];
                if (targetIndex == geneIndex) {
                    targetIndex = -1;  // Already in place
                } else if (targetIndex >= 0) {
                    movedGene = genome->genes[geneIndex];
                }
            }
            block.sync();
            if (targetIndex >= 0) {
                genome->genes[targetIndex] = movedGene;
            }
            block.sync();
        }

        if (laneId == 0) {
            genome->numGenes = newNumGenes;
        }
    }
    block.sync();
}

__inline__ __device__ void GeneGraphProcessor::removeCyclesNotThroughRoot(SimulationData& data, Genome* genome)
{
    // The genes form a directed graph whose edges are the constructor references of their nodes. A cycle through the root gene is
    // intended (constructing the root gene starts a new creature), while a cycle avoiding the root gene means unbounded
    // construction within the same creature. Every cycle either contains the root gene or lies entirely within the remaining
    // genes, so the requirement is exactly that the subgraph induced on the genes 1 .. numGenes - 1 is acyclic.
    //
    // A depth-first search over that subgraph detects and repairs this in one linear pass: a cycle exists if and only if the
    // search finds a back edge, i.e. an edge onto a gene that is still on the DFS stack. Turning off the constructor of every
    // back edge as it is found leaves an acyclic graph and only sacrifices the constructors that actually close a cycle.
    auto block = cg_geneGraph::this_thread_block();
    auto laneId = block.thread_rank();

    __shared__ int numGenes;
    if (laneId == 0) {
        numGenes = genome->numGenes;
    }
    block.sync();
    if (numGenes <= 1) {  // Uniform across the block, so the early return does not desync the cooperative group
        return;
    }

    __shared__ int* state;  // 0 = not visited, 1 = on the DFS stack, 2 = finished
    __shared__ int* stackGenes;
    __shared__ int* stackNodeIndices;  // Next node of the gene on that stack level that still needs to be examined

    if (laneId == 0) {
        state = data.entities.heap.getTypedSubArray<int>(numGenes);
        // A gene is pushed at most once because only genes in state 0 are pushed and they are marked immediately, so the DFS
        // stack never holds more than numGenes entries.
        stackGenes = data.entities.heap.getTypedSubArray<int>(numGenes);
        stackNodeIndices = data.entities.heap.getTypedSubArray<int>(numGenes);
    }
    block.sync();

    for (int geneIndex = laneId; geneIndex < numGenes; geneIndex += blockDim.x) {
        state[geneIndex] = 0;
    }
    block.sync();

    // The search is inherently sequential, but it visits every gene and every node only once, which is cheaper than the passes
    // that already scan the genome per gene.
    if (laneId == 0) {
        // Marking the root gene as finished takes it out of the graph: edges into it are ignored and cycles through it are kept.
        state[0] = 2;

        for (int startGene = 1; startGene < numGenes; ++startGene) {
            if (state[startGene] != 0) {
                continue;
            }
            state[startGene] = 1;
            stackGenes[0] = startGene;
            stackNodeIndices[0] = 0;
            int stackSize = 1;

            while (stackSize > 0) {
                auto& gene = genome->genes[stackGenes[stackSize - 1]];
                auto nodeIndex = stackNodeIndices[stackSize - 1];
                if (nodeIndex >= gene.numNodes) {
                    state[stackGenes[stackSize - 1]] = 2;
                    --stackSize;
                    continue;
                }
                stackNodeIndices[stackSize - 1] = nodeIndex + 1;

                auto& node = gene.nodes[nodeIndex];
                if (!node.constructorAvailable) {
                    continue;
                }
                auto targetGene = node.constructor.geneIndex;
                if (state[targetGene] == 1) {
                    // Back edge: the target is still on the stack, so this constructor closes a cycle avoiding the root gene.
                    // A gene referencing itself is covered as well, since such a gene is on the stack while it is scanned.
                    node.constructorAvailable = false;
                } else if (state[targetGene] == 0) {
                    state[targetGene] = 1;
                    stackGenes[stackSize] = targetGene;
                    stackNodeIndices[stackSize] = 0;
                    ++stackSize;
                }
                // A finished target is a cross or forward edge and does not close a cycle.
            }
        }
    }
    block.sync();
}
