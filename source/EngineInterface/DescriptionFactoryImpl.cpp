#include "DescriptionFactoryImpl.h"

#include <math.h>

#include <QRandomGenerator>

#include "Physics.h"


ClusterDescription DescriptionFactoryImpl::createHexagon(
    CreateHexagonParameters const& parameters,
    NumberGenerator* numberGenerator) const
{
    auto const layers = parameters._layers;
    std::vector<std::vector<CellDescription>> cellMatrix(2 * layers - 1, std::vector<CellDescription>(2 * layers - 1));
    list<CellDescription> cells;

    double incY = std::sqrt(3.0) * parameters._cellDistance / 2.0;
    for (int j = 0; j < layers; ++j) {
        for (int i = -(layers - 1); i < layers - j; ++i) {

            //create cell: upper layer
            cellMatrix[layers - 1 + i][layers - 1 - j] =
                CellDescription()
                    .setId(numberGenerator->getId())
                    .setVel(parameters._velocity)
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

            //create cell: under layer (except for 0-layer)
            if (j > 0) {
                cellMatrix[layers - 1 + i][layers - 1 + j] =
                    CellDescription()
                        .setId(numberGenerator->getId())
                        .setEnergy(parameters._cellEnergy)
                        .setPos(
                            parameters._centerPosition
                            + QVector2D{static_cast<float>(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), static_cast<float>(+j * incY)})
                        .setVel(parameters._velocity)
                        .setMaxConnections(parameters._maxConnections)
                        .setFlagTokenBlocked(false)
                        .setTokenBranchNumber(0)
                        .setMetadata(CellMetadata().setColor(parameters._colorCode))
                        .setCellFeature(CellFeatureDescription());
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

    auto hexagon = ClusterDescription()
                       .setId(numberGenerator->getId())
                       .setVel({0, 0})
                       .setAngle(0)
                       .setAngularVel(0)
                       .setMetadata(ClusterMetadata())
                       .addCells(cells);
    hexagon.setPos(hexagon.getClusterPosFromCells());

    std::unordered_map<uint64_t, int> cache;
    for (int j = 0; j < layers; ++j) {
        for (int i = -(layers - 1); i < layers - j; ++i) {

            if (layers - 1 + i > 0) {
                hexagon.addConnection(
                    cellMatrix[layers - 1 + i][layers - 1 - j].id, cellMatrix[layers - 1 + i - 1][layers - 1 - j].id, cache);
            }
            if (j > 0) {
                hexagon.addConnection(
                    cellMatrix[layers - 1 + i][layers - 1 - j].id,
                    cellMatrix[layers - 1 + i][layers - 1 - j + 1].id,
                    cache);
                hexagon.addConnection(
                    cellMatrix[layers - 1 + i][layers - 1 - j].id,
                    cellMatrix[layers - 1 + i + 1][layers - 1 - j + 1].id,
                    cache);
            }

            if (j > 0) {
                if (layers - 1 + i > 0) {
                    hexagon.addConnection(
                        cellMatrix[layers - 1 + i][layers - 1 + j].id,
                        cellMatrix[layers - 1 + i - 1][layers - 1 + j].id,
                        cache);
                }
                hexagon.addConnection(
                    cellMatrix[layers - 1 + i][layers - 1 + j].id,
                    cellMatrix[layers - 1 + i][layers - 1 + j - 1].id,
                    cache);
                hexagon.addConnection(
                    cellMatrix[layers - 1 + i][layers - 1 + j].id,
                    cellMatrix[layers - 1 + i + 1][layers - 1 + j - 1].id,
                    cache);
            }
        }
    }

    return hexagon;
}

ClusterDescription DescriptionFactoryImpl::createRect(
    CreateRectParameters const& parameters,
    NumberGenerator* numberGenerator) const
{
    auto size = parameters._size;
    auto distance = parameters._cellDistance;
    auto energy = parameters._cellEnergy;
    auto colorCode = parameters._colorCode;

    vector<vector<CellDescription>> cellMatrix;
    for (int x = 0; x < size.x; ++x) {
        vector<CellDescription> cellRow;
        for (int y = 0; y < size.y; ++y) {
            cellRow.push_back(CellDescription()
                                  .setId(numberGenerator->getId())
                                  .setEnergy(energy)
                                  .setPos({static_cast<float>(x), static_cast<float>(y)})
                                  .setMaxConnections(parameters._maxConnections)
                                  .setFlagTokenBlocked(false)
                                  .setTokenBranchNumber(0)
                                  .setMetadata(CellMetadata().setColor(colorCode))
                                  .setCellFeature(CellFeatureDescription()));
        }
        cellMatrix.push_back(cellRow);
    }
    auto result = ClusterDescription()
                       .setId(numberGenerator->getId())
                       .setPos({static_cast<float>(size.x) / 2.0f - 0.5f, static_cast<float>(size.y) / 2.0f - 0.5f})
                       .setVel({0, 0})
                       .setAngle(0)
                       .setAngularVel(0)
                       .setMetadata(ClusterMetadata());
    for (int x = 0; x < size.x; ++x) {
        for (int y = 0; y < size.y; ++y) {
            result.addCell(cellMatrix[x][y]);
        }
    }

    std::unordered_map<uint64_t, int> cache;
    for (int x = 0; x < size.x; ++x) {
        for (int y = 0; y < size.y; ++y) {
            if (x > 0) {
                result.addConnection(cellMatrix[x][y].id, cellMatrix[x - 1][y].id, cache);
            }
            if (y > 0) {
                result.addConnection(cellMatrix[x][y].id, cellMatrix[x][y - 1].id, cache);
            }
        }
    }

    return result;
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
                for (auto const& connection : *cell.connections) {
                    if (visitedCellIds.find(connection.cellId) == visitedCellIds.end()) {
                        cellIdPath.emplace_back(connection.cellId);
                        visitedCellIds.insert(connection.cellId);
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
}

void DescriptionFactoryImpl::randomizeCellFunctions(
    SimulationParameters const& parameters,
    DataDescription& data,
    std::unordered_set<uint64_t> const& cellIds) const
{

    DescriptionNavigator navigator;
    navigator.update(data);
    for (auto const& cellId : cellIds) {
        auto clusterIndex = navigator.clusterIndicesByCellIds.at(cellId);
        auto cellIndex = navigator.cellIndicesByCellIds.at(cellId);
        auto& cell = data.clusters->at(clusterIndex).cells->at(cellIndex);

        CellFeatureDescription cellFunction;
        cellFunction.setType(static_cast<Enums::CellFunction::Type>(
            QRandomGenerator::global()->generate() % Enums::CellFunction::_COUNTER));

        QByteArray volatileData;
        for (int i = 0; i < parameters.cellFunctionComputerMaxInstructions * 3; ++i) {
            volatileData.append(QRandomGenerator::global()->generate() % 256);
        }
        cellFunction.setVolatileData(volatileData);

        QByteArray staticData;
        for (int i = 0; i < parameters.cellFunctionComputerCellMemorySize; ++i) {
            staticData.append(QRandomGenerator::global()->generate() % 256);
        }
        cellFunction.setConstData(staticData);
        cell.cellFeature = cellFunction;
    }
}

void DescriptionFactoryImpl::removeFreeCellConnections(
    SimulationParameters const& parameters,
    DataDescription& data,
    std::unordered_set<uint64_t> const& cellIds) const
{
    DescriptionNavigator navigator;
    navigator.update(data);
    for (auto const& cellId : cellIds) {
        auto clusterIndex = navigator.clusterIndicesByCellIds.at(cellId);
        auto cellIndex = navigator.cellIndicesByCellIds.at(cellId);
        auto& cell = data.clusters->at(clusterIndex).cells->at(cellIndex);
        cell.maxConnections = cell.connections ? cell.connections->size() : 0;
    }
}
