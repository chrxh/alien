#include "DescriptionFactoryImpl.h"

#include <math.h>

#include "Physics.h"


ClusterDescription DescriptionFactoryImpl::createHexagon(CreateHexagonParameters const& parameters) const
{
    auto addConnection = [](CellDescription& cell1, CellDescription& cell2) {
        cell1.addConnection(cell2.id);
        cell2.addConnection(cell1.id);
    };

    auto const layers = parameters._layers;
    std::vector<std::vector<CellDescription>> cellMatrix(2 * layers - 1, std::vector<CellDescription>(2 * layers - 1));
    list<CellDescription> cells;

    uint64_t id = 0;
    double incY = std::sqrt(3.0) * parameters._cellDistance / 2.0;
    for (int j = 0; j < layers; ++j) {
        for (int i = -(layers - 1); i < layers - j; ++i) {

            //create cell: upper layer
            cellMatrix[layers - 1 + i][layers - 1 - j] =
                CellDescription()
                    .setId(++id)
                    .setEnergy(parameters._cellEnergy)
                    .setPos(
                        parameters._centerPosition
                        + QVector2D{static_cast<float>(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), 
                                    static_cast<float>(-j * incY)})
                    .setMaxConnections(parameters._maxConnections)
                    .setFlagTokenBlocked(false)
                    .setTokenBranchNumber(0)
                    .setMetadata(CellMetadata().setColor(parameters._colorCode))
                    .setCellFeature(CellFeatureDescription());

            if (layers - 1 + i > 0) {
                addConnection(
                    cellMatrix[layers - 1 + i][layers - 1 - j], cellMatrix[layers - 1 + i - 1][layers - 1 - j]);
            }
            if (j > 0) {
                addConnection(
                    cellMatrix[layers - 1 + i][layers - 1 - j], cellMatrix[layers - 1 + i][layers - 1 - j + 1]);
                addConnection(
                    cellMatrix[layers - 1 + i][layers - 1 - j], cellMatrix[layers - 1 + i + 1][layers - 1 - j + 1]);
            }

            //create cell: under layer (except for 0-layer)
            if (j > 0) {
                cellMatrix[layers - 1 + i][layers - 1 + j] =
                    CellDescription()
                        .setId(++id)
                        .setEnergy(parameters._cellEnergy)
                        .setPos(
                            parameters._centerPosition
                            + QVector2D{static_cast<float>(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), static_cast<float>(+j * incY)})
                        .setMaxConnections(parameters._maxConnections)
                        .setFlagTokenBlocked(false)
                        .setTokenBranchNumber(0)
                        .setMetadata(CellMetadata().setColor(parameters._colorCode))
                        .setCellFeature(CellFeatureDescription());

                if (layers - 1 + i > 0) {
                    addConnection(
                        cellMatrix[layers - 1 + i][layers - 1 + j], cellMatrix[layers - 1 + i - 1][layers - 1 + j]);
                }
                addConnection(
                    cellMatrix[layers - 1 + i][layers - 1 + j], cellMatrix[layers - 1 + i][layers - 1 + j - 1]);
                addConnection(
                    cellMatrix[layers - 1 + i][layers - 1 + j], cellMatrix[layers - 1 + i + 1][layers - 1 + j - 1]);
            }
        }
    }

    for (auto const& cellRow : cellMatrix) {
        for (auto const& cell : cellRow) {
            if (cell.id > 0) {
                cells.push_back(cell);
            }
        }
    }

    auto hexagon =
        ClusterDescription().setVel({0, 0}).setAngle(0).setAngularVel(0).setMetadata(ClusterMetadata()).addCells(cells);
    hexagon.setPos(hexagon.getClusterPosFromCells());

    if (parameters._angle != 0) {
        for (auto& cell : *hexagon.cells) {
            cell.pos = Physics::rotateClockwise(*cell.pos - *hexagon.pos, parameters._angle) + *hexagon.pos;
        }
    }

    return hexagon;
}

ClusterDescription DescriptionFactoryImpl::createUnconnectedDisc(
    CreateDiscParameters const& parameters) const
{
    auto circle =
        ClusterDescription().setVel({0, 0}).setAngle(0).setAngularVel(0).setMetadata(ClusterMetadata());

    uint64_t id = 0;
    for (double radius = parameters._innerRadius; radius - FLOATINGPOINT_HIGH_PRECISION <= parameters._outerRadius;
         radius += parameters._cellDistance) {
        auto angleInc = radius > 0 ? asin(parameters._cellDistance / (2.0 * radius)) * 2.0 * radToDeg : 361.0;
        angleInc = 360.0 / floor(360.0 / angleInc);
        std::unordered_set<uint64_t> cellIds;
        for (auto angle = 0.0; angle < 360.0 - angleInc/2; angle += angleInc) {
            auto relPos = Physics::unitVectorOfAngle(angle) * radius;

            auto cell = CellDescription()
                            .setId(++id)
                            .setEnergy(parameters._cellEnergy)
                            .setPos(parameters._centerPosition + relPos)
                            .setMaxConnections(parameters._maxConnections)
                            .setFlagTokenBlocked(false)
                            .setTokenBranchNumber(0)
                            .setMetadata(CellMetadata().setColor(parameters._colorCode))
                            .setCellFeature(CellFeatureDescription());
            circle.addCell(cell);
        }
    }

    circle.setPos(circle.getClusterPosFromCells());

    return circle;
}

void DescriptionFactoryImpl::generateBranchNumbers(
    SimulationParameters const& parameters,
    DataDescription& data,
    std::unordered_set<uint64_t> const& cellIds) const
{
    DescriptionNavigator navigator;
    navigator.update(data);

    std::set<uint64_t> visitedCellIds(cellIds.begin(), cellIds.end());
    std::vector<std::vector<uint64_t>> cellIdPaths;
    for (auto const& cellId : cellIds) {
        cellIdPaths.emplace_back(std::vector<uint64_t>{cellId});
    }

    int origNumVisitedCells = 0;
    do {

        //set branch numbers an last cell on path
        for (auto const& cellIdPath : cellIdPaths) {
            if (cellIdPath.empty()) {
                continue;
            }
            auto const& lastCellId = cellIdPath.back();

            auto clusterIndex = navigator.clusterIndicesByCellIds.at(lastCellId);
            auto cellIndex = navigator.cellIndicesByCellIds.at(lastCellId);
            auto& cell = data.clusters->at(clusterIndex).cells->at(cellIndex);
            cell.setTokenBranchNumber((cellIdPath.size() - 1) % parameters.cellMaxTokenBranchNumber);
        }

        //modify paths
        origNumVisitedCells = visitedCellIds.size();
        for (auto& cellIdPath : cellIdPaths) {
            auto found = false;
            while (!found && !cellIdPath.empty()) {
                auto const& lastCellId = cellIdPath.back();
                auto clusterIndex = navigator.clusterIndicesByCellIds.at(lastCellId);
                auto cellIndex = navigator.cellIndicesByCellIds.at(lastCellId);
                auto& cell = data.clusters->at(clusterIndex).cells->at(cellIndex);
                for (auto const& connectingCellId : *cell.connectingCells) {
                    if (visitedCellIds.find(connectingCellId) == visitedCellIds.end()) {
                        cellIdPath.emplace_back(connectingCellId);
                        visitedCellIds.insert(connectingCellId);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    cellIdPath.pop_back();
                }
            }
        }
    } while (origNumVisitedCells != visitedCellIds.size());

    /*
    std::set<uint64_t> visitedCellIds;
    std::set<uint64_t> currentCellIds(cellIds.begin(), cellIds.end());
    int branchNumber = 0;
    int origNumVisitedCells = 0;
    do {
        std::set<uint64_t> adjacentCellIds;
        for (auto const& cellId : currentCellIds) {
            auto clusterIndex = navigator.clusterIndicesByCellIds.at(cellId);
            auto cellIndex = navigator.cellIndicesByCellIds.at(cellId);
            auto& cell = data.clusters->at(clusterIndex).cells->at(cellIndex);
            cell.setTokenBranchNumber(branchNumber);

            for (auto const& connectingCellId : *cell.connectingCells) {
                if (visitedCellIds.find(connectingCellId) == visitedCellIds.end()
                    && currentCellIds.find(connectingCellId) == currentCellIds.end()) {
                    adjacentCellIds.insert(connectingCellId);
                    break;
                }
            }
        }

        origNumVisitedCells = visitedCellIds.size();
        visitedCellIds.insert(currentCellIds.begin(), currentCellIds.end());

        currentCellIds = adjacentCellIds;
        branchNumber = (branchNumber + 1) % parameters.cellMaxTokenBranchNumber;
    } while (origNumVisitedCells != visitedCellIds.size());
*/
}
