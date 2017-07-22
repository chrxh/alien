#pragma once

#include "Model/Entities/Descriptions.h"

#include "Gui/Definitions.h"

class VisualDescription
	: public QObject
{
	Q_OBJECT
public:
	VisualDescription(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~VisualDescription() = default;

	virtual DataDescription& getDataRef();
	virtual map<uint64_t, CellDescription> getCellDescsByIds() const;
	virtual void setData(DataDescription const &data);
	virtual void setSelection(uint64_t cellIds, uint64_t particleIds);

private:
	DataDescription _data;
};
