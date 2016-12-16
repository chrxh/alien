#ifndef SERIALIZATIONFACADEIMPL_H
#define SERIALIZATIONFACADEIMPL_H

#include "model/serializationfacade.h"

class SerializationFacadeImpl : public SerializationFacade
{
public:
	SerializationFacadeImpl();
	~SerializationFacadeImpl() = default;

	void serializeSimulationContext(SimulationContext* context, QDataStream& stream);
	SimulationContext* deserializeSimulationContext(QDataStream& stream);

	void serializeCellCluster(CellCluster* cluster, QDataStream& stream);
	CellCluster* deserializeCellCluster(QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap
		, QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap, Grid* grid);

	void serializeFeaturedCell(Cell* cell, QDataStream& stream);
	Cell* deserializeFeaturedCell(QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, Grid* grid);
	Cell* deserializeFeaturedCell(QDataStream& stream, Grid* grid);

};

#endif //SERIALIZATIONFACADEIMPL_H
