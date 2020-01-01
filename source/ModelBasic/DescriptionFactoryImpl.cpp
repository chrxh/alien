#include "Physics.h"

#include "DescriptionFactoryImpl.h"

ClusterDescription DescriptionFactoryImpl::createHexagon(CreateHexagonParameters const& parameters)
{
    auto addConnection = [](CellDescription& cell1, CellDescription& cell2) {
        cell1.addConnection(cell2.id);
        cell2.addConnection(cell1.id);
    };

    auto const layers = parameters._layers;
    std::vector<std::vector<CellDescription>> cellMatrix(2 * layers - 1, std::vector<CellDescription>(2 * layers - 1));
    list<CellDescription> cells;

    int maxCon = 6;
    uint64_t id = 0;
    double incY = std::sqrt(3.0) * parameters._cellDistance / 2.0;
    for (int j = 0; j < layers; ++j) {
        for (int i = -(layers - 1); i < layers - j; ++i) {

            //check if cell is on boundary
            if (((i == -(layers - 1)) || (i == layers - j - 1)) && ((j == 0) || (j == layers - 1))) {
                maxCon = 3;
            } else if ((i == -(layers - 1)) || (i == layers - j - 1) || (j == layers - 1)) {
                maxCon = 4;
            } else {
                maxCon = 6;
            }

            //create cell: upper layer
            cellMatrix[layers - 1 + i][layers - 1 - j] =
                CellDescription()
                    .setId(++id)
                    .setEnergy(parameters._cellEnergy)
                    .setPos(
                        parameters._centerPosition
                        + QVector2D{static_cast<float>(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), 
                                    static_cast<float>(-j * incY)})
                    .setMaxConnections(maxCon)
                    .setFlagTokenBlocked(false)
                    .setTokenBranchNumber(0)
                    .setMetadata(CellMetadata())
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
                            + QVector2D{static_cast<float>(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), 
                                        static_cast<float>(+j * incY)})
                        .setMaxConnections(maxCon)
                        .setFlagTokenBlocked(false)
                        .setTokenBranchNumber(0)
                        .setMetadata(CellMetadata())
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
