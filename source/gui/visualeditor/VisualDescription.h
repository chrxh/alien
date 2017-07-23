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
	virtual void setSelection(vector<uint64_t> const &cellIds, vector<uint64_t> const &particleIds);

private:
	DataDescription _data;
};
