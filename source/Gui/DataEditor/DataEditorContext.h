#pragma once

#include <QObject>

#include "Model/Entities/Descriptions.h"

class DataEditorContext
	: public QObject
{
	Q_OBJECT
public:
	DataEditorContext(QObject *parent = nullptr) : QObject(parent) {}
	virtual ~DataEditorContext() = default;

	void setData(boost::optional<DataDescription> const& value);
	boost::optional<DataDescription> getData();

	Q_SIGNAL void notifyClusterEditor();
	Q_SIGNAL void notifyOthers();

private:
	boost::optional<DataDescription> _clusterUnderEdit;
};