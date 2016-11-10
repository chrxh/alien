#ifndef METADATAMANAGER_H
#define METADATAMANAGER_H

#include <QMap>
#include <QString>

struct CellMetadata
{
    QString computerCode;
//    quint8 colorNumber;
    QString name;
    QString description;
    QString clusterName;
};

class MetadataManager
{
public:
    static MetadataManager& getGlobalInstance ();

    MetadataManager ();
    ~MetadataManager ();

    void setCellCode (quint64 id, QString code);
    QString getCellCode (quint64 id);
//    void setCellColorNumber(quint64 id, quint8 color);
//    quint8 getCellColorNumber (quint64 id);
    void setCellName (quint64 id, QString name);
    QString getCellName (quint64 id);
    void setCellDescription (quint64 id, QString descr);
    QString getCellDescription (quint64 id);
    void setClusterName (quint64 id, QString name);     //id = cellId
    QString getClusterName (quint64 id);                //id = cellId
    void setAndUniteClusterName (const QList< quint64 >& ids, QString clusterName);       //list contains ids of all cells in the cluster
    QString getAndUniteClusterName (const QList< quint64 >& ids);    //list contains ids of all cells in the cluster

    void cleanUp (const QSet< quint64 >& ids);          //preserve metadata for the cells specified in ids

    const QMap< quint64, CellMetadata >& getCellMetadata ();

    void addSymbolEntry (QString key, QString value);
    void delSymbolEntry (QString key);
    QString applySymbolTableToCode (QString input);
    void clearSymbolTable ();
    const QMap< QString, QString >& getSymbolTable ();
    void setSymbolTable (const QMap< QString, QString >& table);

    void serializeMetadataCell (QDataStream& stream, quint64 clusterId, quint64 cellId);
    void serializeMetadataEnsemble (QDataStream& stream, const QList< quint64 >& clusterIds, const QList< quint64 >& cellIds);
    void readMetadata (QDataStream& stream, const QMap< quint64, quint64 >& oldNewClusterIdMap, const QMap< quint64, quint64 >& oldNewCellIdMap);
    void serializeMetadataUniverse (QDataStream& stream);
    void readMetadataUniverse (QDataStream& stream, const QMap< quint64, quint64 >& oldNewClusterIdMap, const QMap< quint64, quint64 >& oldNewCellIdMap);

    void serializeSymbolTable (QDataStream& stream);
    void readSymbolTable (QDataStream& stream, bool merge = false);

private:
    QMap< quint64, CellMetadata > _idCellMetadataMap;
    QMap< QString, QString > _symbolTable;
};

#endif // METADATAMANAGER_H
