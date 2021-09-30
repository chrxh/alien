#include "DescriptionHelper.h"

#include "Base/NumberGenerator.h"

void DescriptionHelper::duplicate(DataDescription& data, IntVector2D const& origSize, IntVector2D const& size)
{
    DataDescription result;

    for (int incX = 0; incX < size.x; incX += origSize.x) {
        for (int incY = 0; incY < size.y; incY += origSize.y) {
            if (data.clusters) {
                for (auto cluster : *data.clusters) {
                    auto origPos = cluster.getClusterPosFromCells();
                    RealVector2D clusterPos = {origPos.x + incX, origPos.y + incY};
                    if (clusterPos.x < size.x && clusterPos.y < size.y) {
                        if (cluster.cells) {
                            for (auto& cell : *cluster.cells) {
                                auto origPos = *cell.pos;
                                cell.pos = RealVector2D{origPos.x + incX, origPos.y + incY};
                            }
                        }
                        makeValid(cluster);
                        result.addCluster(cluster);
                    }
                }
            }
            if (data.particles) {
                for (auto particle : *data.particles) {
                    auto origPos = *particle.pos;
                    particle.pos = RealVector2D{origPos.x + incX, origPos.y + incY};
                    if (particle.pos->x < size.x && particle.pos->y < size.y) {
                        particle.setId(NumberGenerator::getInstance().getId());
                        result.addParticle(particle);
                    }
                }
            }
        }
    }
    data = result;
}

void DescriptionHelper::makeValid(ClusterDescription& cluster)
{
    auto& numberGen = NumberGenerator::getInstance();
    cluster.id = numberGen.getId();
    if (cluster.cells) {
        unordered_map<uint64_t, uint64_t> newByOldIds;
        for (auto& cell : *cluster.cells) {
            uint64_t newId = numberGen.getId();
            newByOldIds.insert_or_assign(cell.id, newId);
            cell.id = newId;
        }

        for (auto& cell : *cluster.cells) {
            if (cell.connections) {
                for (auto& connection : *cell.connections) {
                    connection.cellId = newByOldIds.at(connection.cellId);
                }
            }
        }
    }
}
