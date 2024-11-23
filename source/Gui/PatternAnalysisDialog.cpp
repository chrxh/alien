#include "PatternAnalysisDialog.h"

#include <fstream>

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptors.hpp>

#include <ImFileDialog.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/Descriptions.h"
#include "PersisterInterface/SerializerService.h"
#include "EngineInterface/SimulationFacade.h"

#include "GenericMessageDialog.h"
#include "MainLoopEntityController.h"


void PatternAnalysisDialog::init(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::get().getValue("dialogs.pattern analysis.starting path", path.string());
}

void PatternAnalysisDialog::shutdown()
{
    GlobalSettings::get().setValue("dialogs.pattern analysis.starting path", _startingPath);
}

void PatternAnalysisDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("PatternAnalysisDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        saveRepetitiveActiveClustersToFiles(firstFilename.string());
    }
    ifd::FileDialog::Instance().Close();
}

void PatternAnalysisDialog::show()
{
    ifd::FileDialog::Instance().Save("PatternAnalysisDialog", "Save pattern analysis result", "Analysis result (*.txt){.txt},.*", _startingPath);
}

void PatternAnalysisDialog::saveRepetitiveActiveClustersToFiles(std::string const& filename)
{
    auto const partitionClassDataByDescription = calcPartitionData();

    std::ofstream file;
    file.open(filename, std::ios_base::out);
    if (!file) {
        GenericMessageDialog::get().information("Pattern analysis", "The analysis result could not be saved to the specified file.");
        return;
    }

    int sum = 0;
    std::vector<PartitionClassData> partitionData;
    for (auto const& [analysisDesc, partitionClassData] : partitionClassDataByDescription) {
        if (partitionClassData.numberOfElements > 1) {
            partitionData.emplace_back(partitionClassData);
        }
    }
    std::sort(partitionData.begin(), partitionData.end());

    file << "number of repetitive active cell networks: " << partitionData.size() << std::endl << std::endl;
    for (auto const& [index, partitionClassData] : partitionData | boost::adaptors::reversed | boost::adaptors::indexed(1)) {

        file << "cell network " << index << ": " << partitionClassData.numberOfElements << " exemplars" << std::endl;

        std::stringstream clusterNameStream;
        clusterNameStream << "cell network" << std::setfill('0') << std::setw(6) << index << ".sim";

        std::filesystem::path clusterFilename(filename);
        clusterFilename.remove_filename();
        clusterFilename /= clusterNameStream.str();

        ClusteredDataDescription pattern;
        pattern.clusters = std::vector<ClusterDescription>{partitionClassData.representant};

        SerializerService::get().serializeContentToFile(clusterFilename.string(), pattern);
    }
    file.close();

    std::stringstream messageStream;
    messageStream << partitionData.size() << " repetitive active cell network found. A summary is saved to " << filename << "." << std::endl;
    if (!partitionData.empty()) {
        messageStream << "Representative cell networks are save from `cluster" << std::setfill('0') << std::setw(6) << 0 << ".sim` to `cluster" << std::setfill('0')
                      << std::setw(6) << partitionData.size() - 1 << ".sim`.";
    }
    GenericMessageDialog::get().information("Analysis result", messageStream.str());
}

auto PatternAnalysisDialog::calcPartitionData() const -> std::map<ClusterAnalysisDescription, PartitionClassData>
{
    auto data = _simulationFacade->getClusteredSimulationData();

    std::map<ClusterAnalysisDescription, PartitionClassData> result;

    for (auto const& cluster : data.clusters) {
        auto const clusterAnalysisData = getAnalysisDescription(cluster);
        auto& partitionData = result[clusterAnalysisData];
        if (1 == ++partitionData.numberOfElements) {
            partitionData.representant = cluster;
        }
    }
    return result;
}

auto PatternAnalysisDialog::getAnalysisDescription(ClusterDescription const& cluster) const -> ClusterAnalysisDescription
{
    ClusterAnalysisDescription result;
    std::map<uint64_t, CellAnalysisDescription> cellAnalysisDescById;
    auto insertCellAnalysisDescription = [&cellAnalysisDescById](CellDescription const& cell) {
        if (cellAnalysisDescById.find(cell.id) == cellAnalysisDescById.end()) {
            CellAnalysisDescription result;
            result.maxConnections = cell.maxConnections;
            result.numConnections = cell.connections.size();
            result.constructionState = cell.livingState;
            result.inputExecutionOrderNumber = cell.inputExecutionOrderNumber;
            result.outputBlocked = cell.outputBlocked;
            result.executionOrderNumber = cell.executionOrderNumber;
            result.color = cell.color;
            result.cellFunction = cell.getCellFunctionType();

            cellAnalysisDescById.insert_or_assign(cell.id, result);
        }
    };

    std::map<uint64_t, int> cellDescIndexById;
    for (auto const& [index, cell] : cluster.cells | boost::adaptors::indexed(0)) {
        cellDescIndexById.insert_or_assign(cell.id, toInt(index));
    }

    for (auto const& cell : cluster.cells) {
        insertCellAnalysisDescription(cell);
        for (auto const& connection : cell.connections) {
            auto connectingCellId = connection.cellId;
            insertCellAnalysisDescription(cluster.cells.at(cellDescIndexById.at(connectingCellId)));
            result.connectedCells.insert(std::set<CellAnalysisDescription>{cellAnalysisDescById.at(cell.id), cellAnalysisDescById.at(connectingCellId)});
        }
    }
    return result;
}
