#include "metadatamanager.h"

#include <QDebug>
#include <QDataStream>
#include <set>

MetadataManager& MetadataManager::getGlobalInstance ()
{
    static MetadataManager instance;
    return instance;
}

MetadataManager::MetadataManager()
{

}

MetadataManager::~MetadataManager()
{

}

void MetadataManager::setCellCode (quint64 id, QString code)
{
    _idCellMetadataMap[id].computerCode = code;
}

QString MetadataManager::getCellCode (quint64 id)
{
    if( _idCellMetadataMap.contains(id) )
        return _idCellMetadataMap[id].computerCode;
    else
        return "";
}

void MetadataManager::setCellName (quint64 id, QString name)
{
    _idCellMetadataMap[id].name = name;
}

QString MetadataManager::getCellName (quint64 id)
{
    if( _idCellMetadataMap.contains(id) )
        return _idCellMetadataMap[id].name;
    else
        return "";
}

void MetadataManager::setCellDescription (quint64 id, QString descr)
{
    _idCellMetadataMap[id].description = descr;
}

QString MetadataManager::getCellDescription (quint64 id)
{
    if( _idCellMetadataMap.contains(id) )
        return _idCellMetadataMap[id].description;
    else
        return "";
}

void MetadataManager::setClusterName (quint64 id, QString name)
{
    _idCellMetadataMap[id].clusterName = name;
}

QString MetadataManager::getClusterName (quint64 id)
{
    if( _idCellMetadataMap.contains(id) )
        return _idCellMetadataMap[id].clusterName;
    else
        return "";
}

void MetadataManager::setAndUniteClusterName (const QList< quint64 >& ids, QString clusterName)
{
    foreach(quint64 id, ids) {
        setClusterName(id, clusterName);
    }
}

QString MetadataManager::getAndUniteClusterName (const QList< quint64 >& ids)
{
    //count the names
    QMap< QString, int > namesCount;
    foreach(quint64 id, ids) {
        if( _idCellMetadataMap.contains(id) ) {
            QString clusterName = _idCellMetadataMap[id].clusterName;
            namesCount[clusterName] = namesCount[clusterName]+1;
        }
        else {
            namesCount[QString()] = namesCount[QString()]+1;
        }

    }

    //determine the most counted name
    QString mostCountedName;
    int mostCount = 0;
    QMapIterator< QString, int > it(namesCount);
    while( it.hasNext() ) {
        it.next();
        QString clusterName = it.key();
        int count = it.value();
        if( count > mostCount ) {
            mostCount = count;
            mostCountedName = clusterName;
        }
    }

    //set cluster name to all cells in the cluster
    setAndUniteClusterName(ids, mostCountedName);

    return mostCountedName;
}

void MetadataManager::cleanUp (std::set< quint64 > const & ids)
{
    QMutableMapIterator< quint64, CellMetadata > it(_idCellMetadataMap);
    while(it.hasNext()) {
        it.next();
        quint64 id = it.key();
        if (ids.find(id) == ids.end()) {
            it.remove();
        }
    }
}

const QMap< quint64, CellMetadata >& MetadataManager::getCellMetadata ()
{
    return _idCellMetadataMap;
}

void MetadataManager::addSymbolEntry (QString key, QString value)
{
    _symbolTable[key] = value;
}

void MetadataManager::delSymbolEntry (QString key)
{
    _symbolTable.remove(key);
}

QString MetadataManager::applySymbolTableToCode (QString input)
{
    if( _symbolTable.contains(input) ) {
        return _symbolTable[input];
    }
    return input;
}

void MetadataManager::clearSymbolTable ()
{
    _symbolTable.clear();
}

const QMap< QString, QString >& MetadataManager::getSymbolTable ()
{
    return _symbolTable;
}

void MetadataManager::setSymbolTable (const QMap< QString, QString >& table)
{
    _symbolTable = table;
}

void MetadataManager::serializeMetadataCell (QDataStream& stream, quint64 clusterId, quint64 cellId)
{
    //cell data
    quint32 size = 1;
    stream << size;
    QString code = getCellCode(cellId);
//        quint8 color = getCellColorNumber(cellId);
    QString name = getCellName(cellId);
    QString descr = getCellDescription(cellId);
    QString clusterName = getClusterName(cellId);
    stream << cellId << code << name << descr << clusterName;
}

void MetadataManager::serializeMetadataEnsemble (QDataStream& stream, const QList< quint64 >& clusterIds, const QList< quint64 >& cellIds)
{
    //cell data
    quint32 size = cellIds.size();
    stream << size;
    foreach(quint64 cellId, cellIds) {
        QString code = getCellCode(cellId);
//        quint8 color = getCellColorNumber(cellId);
        QString name = getCellName(cellId);
        QString descr = getCellDescription(cellId);
        QString clusterName = getClusterName(cellId);
        stream << cellId << code << name << descr << clusterName;
    }

    //cluster data
/*    size = clusterIds.size();
    stream << size;
    foreach(quint64 clusterId, clusterIds) {
        stream << clusterId << getClusterName(clusterId);
    }*/
}

void MetadataManager::readMetadata (QDataStream& stream, const QMap< quint64, quint64 >& oldNewClusterIdMap, const QMap< quint64, quint64 >& oldNewCellIdMap)
{
    //cell data
    quint32 size = 0;
    stream >> size;
    quint64 id = 0;
    QString code;
//    quint8 color = 0;
    QString name;
    QString descr;
    QString clusterName;
    for(quint32 i = 0; i < size; ++i) {
        stream >> id >> code >> name >> descr >> clusterName;
        if( oldNewCellIdMap.contains(id) ) {
            setCellCode(oldNewCellIdMap[id], code);
//            setCellColorNumber(oldNewCellIdMap[id], color);
            setCellName(oldNewCellIdMap[id], name);
            setCellDescription(oldNewCellIdMap[id], descr);
            setClusterName(oldNewCellIdMap[id], clusterName);
        }
    }

    //cluster data
/*    stream >> size;
    for(int i = 0; i < size; ++i) {
        stream >> id >> name;
        if( oldNewClusterIdMap.contains(id) ) {
            setClusterName(oldNewClusterIdMap[id], name);
        }
    }*/
}

void MetadataManager::serializeMetadataUniverse (QDataStream& stream)
{
    //cell data
    quint32 size = _idCellMetadataMap.size();
    stream << size;
    QMapIterator< quint64, CellMetadata > it(_idCellMetadataMap);
    while(it.hasNext()) {
        it.next();
        quint64 cellId = it.key();
        CellMetadata cellMeta = it.value();
        stream << cellId << cellMeta.computerCode << cellMeta.name << cellMeta.description << cellMeta.clusterName;
    }

    //cluster data
/*    size = _idClusterMetadataMap.size();
    stream << size;
    QMapIterator< quint64, CellClusterMetadata > it2(_idClusterMetadataMap);
    while(it2.hasNext()) {
        it2.next();
        quint64 clusterId = it2.key();
        CellClusterMetadata clusterMeta = it2.value();
        stream << clusterId << clusterMeta.clusterName;
    }*/
}

void MetadataManager::readMetadataUniverse (QDataStream& stream, const QMap< quint64, quint64 >& oldNewClusterIdMap, const QMap< quint64, quint64 >& oldNewCellIdMap)
{
    _idCellMetadataMap.clear();
    readMetadata(stream, oldNewClusterIdMap, oldNewCellIdMap);
}

void MetadataManager::serializeSymbolTable (QDataStream& stream)
{
    stream << _symbolTable;
}

void MetadataManager::readSymbolTable (QDataStream& stream, bool merge)
{
    if( !merge )
        _symbolTable.clear();
    QMap< QString, QString > newSymbolTable;
    stream >> newSymbolTable;
    QMapIterator< QString, QString > it(newSymbolTable);
    while(it.hasNext()) {
        it.next();
        _symbolTable.insert(it.key(), it.value());
    }
}





