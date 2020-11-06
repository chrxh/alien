#include "WebSimulationTableModel.h"

#include <QBrush>
#include <QString>
#include <QVariant>

int WebSimulationTableModel::rowCount(const QModelIndex &parent /*= QModelIndex()*/) const
{
    return _simulationInfos.size();
}

int WebSimulationTableModel::columnCount(const QModelIndex &parent /*= QModelIndex()*/) const
{
    return 5;
}

QVariant WebSimulationTableModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid()) {
        return QVariant();
    }

    if (role == Qt::DisplayRole) {
        if (index.row() >= _simulationInfos.size()) {
            return QVariant();
        }
        auto const simulationInfo = _simulationInfos.at(index.row());
        switch (index.column()) {
        case 0:
            return simulationInfo.isActive;
        case 1:
            return QString::fromStdString(simulationInfo.simulationName);
        case 2:
            return QString::fromStdString(simulationInfo.userName);
        case 3:
            return QString("%1 x %2").arg(simulationInfo.worldSize.x).arg(simulationInfo.worldSize.y);
        case 4:
            return simulationInfo.timestep;
        default:
        return QVariant();
        }
    }
    else {
        return QVariant();
    }
}

QVariant WebSimulationTableModel::headerData(int section, Qt::Orientation orientation, int role /*= Qt::DisplayRole*/) const
{
    if (role == Qt::DisplayRole) {
        if (Qt::Orientation::Horizontal == orientation) {
            switch (section) {
            case 0:
                return QString("Active");
            case 1:
                return QString("Simulation name");
            case 2:
                return QString("User name");
            case 3:
                return QString("World size");
            case 4:
                return QString("Time step");
            default:
                return QString();
            }
        }
    }
    return QVariant();
}

void WebSimulationTableModel::setSimulationInfos(vector<SimulationInfo> const & value)
{
    beginResetModel();
    _simulationInfos = value;
    endResetModel();
}
