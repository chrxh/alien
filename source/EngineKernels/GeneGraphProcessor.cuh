#pragma once

#include <cooperative_groups.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/ShapeGenerator.h>

#include "SimulationData.cuh"

namespace cg_geneGraph = cooperative_groups;

// Operations on the graphs of a genome: the gene graph, whose nodes are the genes and whose edges are the constructor references
// of their nodes, and the node graph of a single gene, whose edges are the connections arising during its construction.
class GeneGraphProcessor
{
public:
    __inline__ __device__ static void voidNodesUnreachableFromLastNode(SimulationData& data, Genome* genome);
    __inline__ __device__ static void removeUnreachableGenesFromRoot(SimulationData& data, Genome* genome);
    __inline__ __device__ static void removeCyclesNotThroughRoot(SimulationData& data, Genome* genome);

private:
    __inline__ __device__ static bool voidUnreachableNodes(SimulationData& data, Gene& gene);
    __inline__ __device__ static void removeMarkedGenes(Genome* genome, int* newGeneIndices);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void GeneGraphProcessor::voidNodesUnreachableFromLastNode(SimulationData& data, Genome* genome)
{
    // A construction remains attached to its constructor at the last constructed node, so everything that is not connected to
    // the last node of a gene is lost. Such nodes are set to void; if the first node of a gene is voided this way, the whole
    // gene is removed from the genome.
    auto block = cg_geneGraph::this_thread_block();
    auto laneId = block.thread_rank();

    __shared__ int numGenes;
    __shared__ int* newGeneIndices;  // -1 = gene is removed

    if (laneId == 0) {
        numGenes = genome->numGenes;
        newGeneIndices = data.entities.heap.getTypedSubArray<int>(genome->numGenes);
    }
    block.sync();

    // The nodes of a gene have to be examined in construction order, so the genes are distributed over the threads.
    for (int geneIndex = laneId; geneIndex < numGenes; geneIndex += blockDim.x) {
        newGeneIndices[geneIndex] = voidUnreachableNodes(data, genome->genes[geneIndex]) ? -1 : 0;
    }
    block.sync();

    if (laneId == 0) {
        // Keep at least one gene so that the genome never becomes empty (as in MutationProcessor::applyMutations_deleteGene).
        bool anySurvivor = false;
        for (int geneIndex = 0; geneIndex < numGenes; ++geneIndex) {
            if (newGeneIndices[geneIndex] >= 0) {
                anySurvivor = true;
                break;
            }
        }
        if (!anySurvivor && numGenes > 0) {
            newGeneIndices[0] = 0;
        }
    }
    block.sync();

    removeMarkedGenes(genome, newGeneIndices);
}

__inline__ __device__ bool GeneGraphProcessor::voidUnreachableNodes(SimulationData& data, Gene& gene)
{
    // With a homogeneous cell type the effective cell type of every node is taken from the first node, so no node is actually
    // void (see EntityFactory::createCellFromNode).
    if (gene.homogeneousCellType || gene.numNodes < 2) {
        return false;
    }

    auto isVoid = [&](int nodeIndex) { return gene.nodes[nodeIndex].cellType == CellType_Void; };

    // Without a void node every node is already connected to the last one via the chain of predecessor connections.
    bool anyVoidNode = false;
    for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
        if (isVoid(nodeIndex)) {
            anyVoidNode = true;
            break;
        }
    }
    if (!anyVoidNode) {
        return false;
    }

    // The connections between the nodes are collected in a union-find structure, so that each of its sets ends up being a
    // connected component of the node graph.
    auto components = data.entities.heap.getTypedSubArray<int>(gene.numNodes);
    for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
        components[nodeIndex] = nodeIndex;
    }
    auto findComponent = [&](int nodeIndex) {
        while (components[nodeIndex] != nodeIndex) {
            components[nodeIndex] = components[components[nodeIndex]];
            nodeIndex = components[nodeIndex];
        }
        return nodeIndex;
    };
    auto connect = [&](int nodeIndex1, int nodeIndex2) {
        auto component1 = findComponent(nodeIndex1);
        auto component2 = findComponent(nodeIndex2);
        if (component1 != component2) {
            components[component2] = component1;
        }
    };

    // Simulate the construction of the gene: a node is connected to its predecessor and to the already constructed nodes
    // reported by the shape generator. A connection only holds if none of its two nodes is void, because a void cell dies right
    // after its construction (see VoidProcessor).
    ShapeGenerator shapeGenerator;
    for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
        auto shapeResult = shapeGenerator.generateNextConstructionData(gene.shape);
        if (isVoid(nodeIndex)) {
            continue;
        }
        if (nodeIndex > 0 && !isVoid(nodeIndex - 1)) {
            connect(nodeIndex, nodeIndex - 1);
        }
        for (int i = 0; i < shapeResult.numAdditionalConnections; ++i) {
            auto otherNodeIndex = shapeResult.requiredNodeId[i];
            if (otherNodeIndex >= 0 && otherNodeIndex < nodeIndex && !isVoid(otherNodeIndex)) {
                connect(nodeIndex, otherNodeIndex);
            }
        }
    }

    auto lastComponent = findComponent(gene.numNodes - 1);
    for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
        if (findComponent(nodeIndex) == lastComponent) {
            continue;
        }
        auto& node = gene.nodes[nodeIndex];
        node.cellType = CellType_Void;
        node.cellTypeData.voidCell = {};
        node.constructorAvailable = false;
    }
    return isVoid(0);
}

__inline__ __device__ void GeneGraphProcessor::removeUnreachableGenesFromRoot(SimulationData& data, Genome* genome)
{
    auto block = cg_geneGraph::this_thread_block();
    auto laneId = block.thread_rank();

    __shared__ int numGenes;
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

    removeMarkedGenes(genome, newGeneIndices);
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

__inline__ __device__ void GeneGraphProcessor::removeMarkedGenes(Genome* genome, int* newGeneIndices)
{
    // Removes every gene marked with a negative value in newGeneIndices; newGeneIndices is overwritten with the new index of
    // each surviving gene (-1 for a removed one). A constructor referencing a removed gene is turned off and an injector
    // referencing one falls back to the first gene.
    auto block = cg_geneGraph::this_thread_block();
    auto laneId = block.thread_rank();

    __shared__ int numGenes;
    __shared__ int newNumGenes;

    if (laneId == 0) {
        numGenes = genome->numGenes;

        // Assign compacted indices to the surviving genes.
        int nextIndex = 0;
        for (int geneIndex = 0; geneIndex < numGenes; ++geneIndex) {
            newGeneIndices[geneIndex] = newGeneIndices[geneIndex] >= 0 ? nextIndex++ : -1;
        }
        newNumGenes = nextIndex;
    }
    block.sync();
    if (newNumGenes == numGenes) {  // Uniform across the block, so the early return does not desync the cooperative group
        return;
    }

    // Remap the references of the surviving genes in parallel.
    for (int geneIndex = laneId; geneIndex < numGenes; geneIndex += blockDim.x) {
        if (newGeneIndices[geneIndex] < 0) {
            continue;
        }
        auto& gene = genome->genes[geneIndex];
        for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
            auto& node = gene.nodes[nodeIndex];
            if (node.constructorAvailable) {
                int mapped = newGeneIndices[node.constructor.geneIndex];
                if (mapped < 0) {
                    node.constructorAvailable = false;
                } else {
                    node.constructor.geneIndex = mapped;
                }
            }
            if (node.cellType == CellType_Injector) {
                int mapped = newGeneIndices[node.cellTypeData.injector.geneIndex];
                node.cellTypeData.injector.geneIndex = mapped < 0 ? 0 : mapped;
            }
        }
    }
    block.sync();

    // Compact the survivors within the existing gene array in parallel. A survivor only moves towards a smaller index and the
    // genes are processed in windows of blockDim.x with a barrier between reading and writing, so a gene is never overwritten
    // before it was read.
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
    block.sync();
}
