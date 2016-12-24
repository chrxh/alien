#ifndef SERIALIZATIONFACADE_H
#define SERIALIZATIONFACADE_H

#include "definitions.h"

class SerializationFacade
{
public:
	virtual ~SerializationFacade() {}

	virtual void serializeSimulationContext(SimulationContext* context, QDataStream& stream) = 0;
	virtual SimulationContext* deserializeSimulationContext(QDataStream& stream) = 0;

	virtual void serializeCellCluster(CellCluster* cluster, QDataStream& stream) = 0;
	virtual CellCluster* deserializeCellCluster(QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap
		, QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap, SimulationContext* context) = 0;

	virtual void serializeFeaturedCell(Cell* cell, QDataStream& stream) = 0;
    virtual Cell* deserializeFeaturedCell(QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, SimulationContext* context) = 0;
    virtual Cell* deserializeFeaturedCell(QDataStream& stream, SimulationContext* context) = 0;
};

#endif //SERIALIZATIONFACADE_H


