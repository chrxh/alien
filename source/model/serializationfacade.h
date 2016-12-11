#ifndef SERIALIZATIONFACADE_H
#define SERIALIZATIONFACADE_H

#include "definitions.h"

class SerializationFacade
{
public:
	virtual ~SerializationFacade() {}

	virtual SimulationContext* deserializeSimulationContext(QDataStream& stream) = 0;
	virtual void serializeSimulationContext(SimulationContext* context, QDataStream& stream) = 0;

	virtual CellCluster* deserializeCellCluster(QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap
		, QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap, Grid* grid) = 0;
	virtual Cell* deserializeFeaturedCell(QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, Grid* grid) = 0;
	virtual Cell* deserializeFeaturedCell(QDataStream& stream, Grid* grid) = 0;
	virtual void serializeFeaturedCell(Cell* cell, QDataStream& stream) = 0;
};

#endif //SERIALIZATIONFACADE_H


