#pragma once

#include "Model/Entities/Descriptions.h"

#include "Gui/Definitions.h"

class DescriptionManager
	: public QObject
{
	Q_OBJECT
public:
	DescriptionManager(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~DescriptionManager() = default;

	virtual DataDescription& getDataRef();
	virtual void setData(DataDescription const &data);
	virtual void changeCellDescription(uint64_t clusterId, CellDescription const &cell);
	virtual CellDescription getCellDescription(uint64_t cellId) const;

private:
	DataDescription _data;
};
